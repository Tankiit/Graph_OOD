"""Vision integration uses generated image fixtures, never benchmark claims."""
import json
from pathlib import Path

import numpy as np
import pytest

from steering_ood.core import ROLES, load_cache
from steering_ood.vision_data import prepare_folders, prepare_builtin


def make_folders(root):
    from PIL import Image
    rng=np.random.default_rng(301)
    for name,n in [('train',24),('test',4),('ood_a',4),('ood_b',4)]:
        for c in range(2):
            p=root/name/f'class{c}';p.mkdir(parents=True)
            for i in range(n):
                pixels=rng.integers(0,256,(32,32,3),dtype='uint8')
                # Distinct images with an easy fixture-only class cue.
                pixels[:,:,c]=pixels[:,:,c]//2+128
                Image.fromarray(pixels).save(p/f'{i:03d}.png')
    return root


def fixture_manifest(tmp_path):
    folders=make_folders(tmp_path/'images')
    manifest=tmp_path/'splits.json'
    prepare_folders(folders/'train',folders/'test',
        [f'a={folders}/ood_a',f'b={folders}/ood_b'],manifest,
        counts=(3,3,3,12,3),seed=3)
    return manifest


def test_image_split_exclusivity_and_groups(tmp_path):
    m=json.loads(fixture_manifest(tmp_path).read_text())
    assert set(r['role'] for r in m['rows'])==set(ROLES)
    assert len(set(r['id'] for r in m['rows']))==len(m['rows'])
    assert len(set(r['pixel_sha256'] for r in m['rows']))==len(m['rows'])
    assert all(r['source']=='train' for r in m['rows'] if r['role'] in ROLES[:5])
    assert all(r['label']==-1 for r in m['rows'] if r['role']=='ood_test')
    assert m['metadata']['ood_groups']==['a','b']


def test_same_dataset_cannot_be_ood(tmp_path):
    with pytest.raises(ValueError,match='distinct'):
        prepare_builtin(tmp_path,tmp_path/'m.json',ood_datasets=['cifar10'])


def test_class_mapping_mismatch(tmp_path):
    folders=make_folders(tmp_path/'images')
    (folders/'test/class1').rename(folders/'test/different_class')
    with pytest.raises(ValueError,match='class names'):
        prepare_folders(folders/'train',folders/'test',[f'a={folders}/ood_a'],tmp_path/'m.json',counts=(2,2,2,2,2))


def test_duplicate_images_are_removed(tmp_path):
    import shutil
    folders=make_folders(tmp_path/'images')
    shutil.copyfile(folders/'test/class0/000.png',folders/'ood_a/class0/000.png')
    m=tmp_path/'m.json'
    prepare_folders(folders/'train',folders/'test',[f'a={folders}/ood_a'],m,counts=(2,2,2,2,2))
    assert len(json.loads(m.read_text())['metadata']['dropped_duplicates'])==2


def test_actual_image_encoder_to_crossed_runner(tmp_path):
    torch=pytest.importorskip('torch')
    from torchvision.transforms import Compose,Resize,ToTensor
    from steering_ood.vision import export_image_cache
    from steering_ood.data import project_cache
    from steering_ood.head import train_head,load_head
    from steering_ood.experiment import Config,evaluate,crossed
    manifest=fixture_manifest(tmp_path)
    encoder=torch.nn.Sequential(torch.nn.Conv2d(3,4,3),torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d((2,2)),torch.nn.Flatten())
    cache=tmp_path/'features.npz'
    export_image_cache(manifest,cache,encoder,Compose([Resize((32,32)),ToTensor()]),
                       {'model':'test CNN','pretrained':False},batch_size=8)
    data=load_cache(cache)
    assert data['reference_x'].shape==(6,16)
    assert data['metadata']['modality']=='vision'
    assert set(data['ood_test_groups'])=={'a','b'}
    projected=tmp_path/'projected.npz';project_cache(cache,projected,4)
    train_head(projected,tmp_path/'head',epochs=2)
    head=load_head(tmp_path/'head',projected)
    cfg=Config(references=2,directions=2,probes=2,steps=9,refine=2,tau=.2)
    result=evaluate(projected,tmp_path/'static',cfg,head)
    assert 'id_classification_accuracy' in result
    for name in ('knn','mahalanobis','energy_T1','msp_T1'):
        assert set(result[name]['ood_by_dataset'])=={'a','b'}
    crossed(projected,tmp_path/'crossed',cfg,head)
    with np.load(tmp_path/'crossed/knn_traces.npz') as f:
        assert f['scores'].shape==(2,2,2,2,9)


@pytest.mark.parametrize('backend,model,dim',[
    ('torchvision','resnet18',512),('timm','vit_tiny_patch16_224',192)])
def test_real_backbone_feature_shape_and_frozen_state(backend,model,dim):
    torch=pytest.importorskip('torch');pytest.importorskip('timm')
    from PIL import Image
    from steering_ood.vision import build_encoder
    net,transform,meta=build_encoder(backend,model,weights='none')
    image=Image.fromarray(np.zeros((40,40,3),dtype='uint8'))
    with torch.inference_mode():
        z=net(transform(image).unsqueeze(0))
    assert tuple(z.shape)==(1,dim)
    assert not net.training and not any(p.requires_grad for p in net.parameters())
    assert not meta['pretrained'] and len(meta['state_sha256'])==64


def test_changed_image_is_rejected(tmp_path):
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor
    from steering_ood.vision import export_image_cache
    manifest=fixture_manifest(tmp_path)
    m=json.loads(manifest.read_text())
    # Modify all selected training sources after the manifest has been hashed.
    for path in (tmp_path/'images/train').rglob('*.png'):
        Image.fromarray(np.zeros((32,32,3),dtype='uint8')).save(path)
    with pytest.raises(ValueError,match='Source image changed'):
        export_image_cache(manifest,tmp_path/'x.npz',torch.nn.Flatten(),ToTensor(),{})


def test_custom_feature_map_requires_explicit_pooling(tmp_path):
    import torch
    from torchvision.transforms import ToTensor
    from steering_ood.vision import export_image_cache
    manifest=fixture_manifest(tmp_path)
    with pytest.raises(ValueError,match='pooling explicitly'):
        export_image_cache(manifest,tmp_path/'x.npz',torch.nn.Identity(),ToTensor(),{})


def test_import_external_cache(tmp_path):
    from steering_ood.data import synthetic_cache,import_features
    p=tmp_path/'raw.npz';synthetic_cache(p)
    metadata=tmp_path/'metadata.json'
    metadata.write_text(json.dumps(dict(modality='vision',encoder='user cached encoder',
                                       split_provenance='fixture IDs assigned before fitting')))
    import_features(p,metadata,tmp_path/'imported.npz')
    assert load_cache(tmp_path/'imported.npz')['metadata']['modality']=='vision'
