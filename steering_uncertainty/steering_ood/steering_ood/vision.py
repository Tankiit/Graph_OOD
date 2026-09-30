"""Frozen image encoders emit the same [N,D] cache contract as NLP encoders."""
import hashlib
import json
from pathlib import Path

import numpy as np
from .core import ROLES, file_hash, save_cache, versions
from .vision_data import open_dataset, image_hash


def state_hash(model):
    import torch
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(t.shape).encode()); h.update(str(t.dtype).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def build_encoder(backend='torchvision',model_name='resnet18',weights='DEFAULT',seed=7,device='cpu'):
    """Return frozen encoder, deterministic RGB preprocessing and provenance.

    torchvision supports ResNet-family and ViT explicitly; timm provides other
    CNNs/transformers through num_classes=0. No classification logits are used as
    features. Only model outputs shaped [B,D] are accepted by the cache exporter.
    """
    import torch
    torch.manual_seed(seed)
    pretrained=weights.lower()!='none'
    if backend=='torchvision':
        from torchvision.models import get_model,get_model_weights
        if not model_name.startswith(('resnet','resnext','wide_resnet','vit_')):
            raise ValueError('Torchvision adapter supports ResNet-family and ViT; use timm for other backbones')
        enum=get_model_weights(model_name)
        chosen=enum.DEFAULT if weights=='DEFAULT' else (None if not pretrained else enum[weights])
        net=get_model(model_name,weights=chosen)
        if model_name.startswith('vit_'):
            net.heads=torch.nn.Identity(); pooling='native CLS token before classification head'
        else:
            net.fc=torch.nn.Identity(); pooling='native global average pool before classification head'
        transform=(chosen or enum.DEFAULT).transforms()
        details=dict(weights=str(chosen),pretrained=pretrained,checkpoint_url=getattr(chosen,'url',None),pooling=pooling)
    elif backend=='timm':
        import timm
        from timm.data import resolve_model_data_config, create_transform
        if weights not in ('DEFAULT','none'):
            raise ValueError('For timm select a tagged model name and use weights DEFAULT or none')
        net=timm.create_model(model_name,pretrained=pretrained,num_classes=0)
        data_config=resolve_model_data_config(net)
        transform=create_transform(**data_config,is_training=False)
        details=dict(weights='tagged timm pretrained config' if pretrained else None,
                     pretrained=pretrained,pretrained_config=net.pretrained_cfg,data_config=data_config,
                     pooling='native timm pooled output with num_classes=0')
    else:
        raise ValueError('backend must be torchvision or timm')
    net.eval().requires_grad_(False)
    details.update(backend=backend,model=model_name,seed=seed,preprocess=repr(transform),
                   state_sha256=state_hash(net),device=device,versions=versions())
    return net.to(device),transform,details


def export_image_cache(manifest_path,output,encoder,preprocess,encoder_metadata,
                       batch_size=64,device='cpu'):
    """Public adapter for any user-owned nn.Module returning [B,D] pooled features.

    Does not guess pooling for dictionaries, token sequences or feature maps.
    Supply a wrapper that explicitly chooses CLS/mean/spatial pooling in that case.
    """
    import torch
    if Path(output).exists():
        raise FileExistsError(output)
    if batch_size<1:
        raise ValueError('batch_size must be positive')
    manifest=json.loads(Path(manifest_path).read_text())
    sources={k:open_dataset(v,download=False) for k,v in manifest['sources'].items()}
    encoder=encoder.to(device).eval().requires_grad_(False)
    arrays={}; feature_dim=None
    ids=[r['id'] for r in manifest['rows']]
    pixels=[r['pixel_sha256'] for r in manifest['rows']]
    if len(set(ids))!=len(ids) or len(set(pixels))!=len(pixels):
        raise ValueError('Image manifest contains repeated IDs or pixels')
    for role in ROLES:
        rows=[r for r in manifest['rows'] if r['role']==role]
        if not rows:
            raise ValueError(f'Empty role {role}')
        batches=[]
        for start in range(0,len(rows),batch_size):
            images=[]
            for row in rows[start:start+batch_size]:
                image,_=sources[row['source']][row['index']]
                image=image.convert('RGB')
                if image_hash(image)!=row['pixel_sha256']:
                    raise ValueError(f"Source image changed after split preparation: {row['id']}")
                images.append(preprocess(image))
            with torch.inference_mode():
                z=encoder(torch.stack(images).to(device))
            if not isinstance(z,torch.Tensor) or z.ndim!=2 or z.shape[0]!=len(images):
                raise ValueError('Encoder must return [batch, feature_dimension]; wrap pooling explicitly')
            if feature_dim is not None and z.shape[1]!=feature_dim:
                raise ValueError('Encoder changed feature dimension')
            feature_dim=z.shape[1]
            batches.append(z.detach().cpu().float().numpy())
        arrays[f'{role}_x']=np.concatenate(batches)
        arrays[f'{role}_y']=np.array([r['label'] for r in rows],dtype='int64')
        arrays[f'{role}_ids']=np.array([r['id'] for r in rows])
        if role=='ood_test':
            arrays['ood_test_groups']=np.array([r['ood_group'] for r in rows])
    # Hash the actual module as well as declared metadata for custom encoders.
    meta=dict(manifest['metadata'],modality='vision',encoder=encoder_metadata,
              actual_encoder_state_sha256=state_hash(encoder),preprocess=repr(preprocess),
              manifest_sha256=file_hash(manifest_path),feature_dim=feature_dim,
              feature_normalization='none after encoder; no projection of path points',
              path_units='Euclidean distance in exported feature coordinates')
    save_cache(output,arrays,meta)


def encode_vision(manifest,output,backend='torchvision',model='resnet18',weights='DEFAULT',
                  batch_size=64,device='cpu',seed=7):
    net,transform,metadata=build_encoder(backend,model,weights,seed,device)
    export_image_cache(manifest,output,net,transform,metadata,batch_size,device)
