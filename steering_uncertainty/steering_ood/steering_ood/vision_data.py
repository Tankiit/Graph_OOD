"""Manifest-based image datasets with disjoint ID roles and OOD evaluation groups."""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from .core import ROLES, write_json

TRAIN_ROLES = ('head', 'reference', 'direction', 'calibration', 'probe')


def image_hash(image):
    """Hash decoded RGB pixels, not filenames; catch duplicate content across roles."""
    image = image.convert('RGB')
    return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()


def open_dataset(spec, download=False):
    from torchvision import datasets
    name, root, split = spec['name'], spec['root'], spec['split']
    if name == 'cifar10':
        return datasets.CIFAR10(root, train=split=='train', download=download)
    if name == 'cifar100':
        return datasets.CIFAR100(root, train=split=='train', download=download)
    if name == 'svhn':
        return datasets.SVHN(root, split=split, download=download)
    if name == 'imagefolder':
        return datasets.ImageFolder(root)
    raise ValueError(f'Unsupported dataset {name}; use ImageFolder or export a feature cache')


def labels_of(dataset):
    if hasattr(dataset, 'targets'):
        return np.asarray(dataset.targets, dtype='int64')
    if hasattr(dataset, 'labels'):
        return np.asarray(dataset.labels, dtype='int64')
    raise ValueError('Dataset must expose targets or labels')


def _identity(spec, index):
    # Include absolute source identity, not just train/test label.
    source = f"{spec['name']}:{spec['root']}:{spec['split']}"
    return f"image:{hashlib.sha256(source.encode()).hexdigest()[:16]}:{index}"


def build_manifest(sources, output, counts=(500,100,100,100,100), seed=7,
                   test_limit=None, download=False):
    """sources maps train/id_test and named ood:* keys to dataset descriptors.

    Training roles are stratified, disjoint and drawn only from the ID train set.
    ImageFolder train/test class names must match, even if integer labels happen
    to coincide. Test capping is deterministic, uniform and independent of scores.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    if len(counts)!=5 or min(counts)<2 or (test_limit is not None and test_limit<1):
        raise ValueError('Need five counts >=2 and a positive optional test limit')
    if 'train' not in sources or 'id_test' not in sources or not any(k.startswith('ood:') for k in sources):
        raise ValueError('Provide ID train, ID test and at least one OOD source')
    datasets = {key:open_dataset(spec,download) for key,spec in sources.items()}
    train, test = datasets['train'], datasets['id_test']
    if hasattr(train,'class_to_idx') and getattr(test,'class_to_idx',None)!=train.class_to_idx:
        raise ValueError('ID train/test ImageFolder class names or mappings differ')
    yt, yi = labels_of(train), labels_of(test)
    classes=np.unique(yt)
    if not set(np.unique(yi))<=set(classes):
        raise ValueError('ID test contains a class absent from ID train')
    mapping={int(c):i for i,c in enumerate(classes)}
    rng=np.random.default_rng(seed); rows=[]

    def add(key, index, role):
        image, raw_label = datasets[key][index]
        spec=sources[key]
        row=dict(id=_identity(spec,index),source=key,index=int(index),role=role,
                 label=-1 if role=='ood_test' else mapping[int(raw_label)],
                 pixel_sha256=image_hash(image))
        if role=='ood_test':
            row['ood_group']=key[4:]
        rows.append(row)

    for c in classes:
        idx=np.flatnonzero(yt==c); rng.shuffle(idx)
        if len(idx)<sum(counts):
            raise ValueError(f'ID class {c} has {len(idx)} images but counts require {sum(counts)}')
        offset=0
        for role,n in zip(TRAIN_ROLES,counts):
            for index in idx[offset:offset+n]:
                add('train',int(index),role)
            offset+=n
    for key in ['id_test']+sorted(k for k in sources if k.startswith('ood:')):
        idx=np.arange(len(datasets[key]))
        if test_limit is not None and len(idx)>test_limit:
            idx=rng.choice(idx,test_limit,replace=False)
        for index in idx:
            add(key,int(index),'id_test' if key=='id_test' else 'ood_test')
    freq=Counter(r['pixel_sha256'] for r in rows)
    dropped=[dict(id=r['id'],role=r['role'],reason='all copies of repeated RGB pixels excluded')
             for r in rows if freq[r['pixel_sha256']]>1]
    rows=[r for r in rows if freq[r['pixel_sha256']]==1]
    for role in TRAIN_ROLES:
        if {r['label'] for r in rows if r['role']==role}!=set(mapping.values()):
            raise ValueError(f'Deduplication removed an entire class from {role}')
    if set(r['role'] for r in rows)!=set(ROLES):
        raise ValueError('A role is empty after duplicate filtering')
    if {r.get('ood_group') for r in rows if r['role']=='ood_test'}!={k[4:] for k in sources if k.startswith('ood:')}:
        raise ValueError('An OOD source is empty after duplicate filtering')
    write_json(output,dict(sources=sources,rows=rows,metadata=dict(modality='vision',seed=seed,
        classes=getattr(train,'classes',[str(c) for c in classes]),class_mapping=mapping,
        counts_per_class_requested=dict(zip(TRAIN_ROLES,counts)),test_limit_per_source=test_limit,
        role_counts=dict(Counter(r['role'] for r in rows)),dropped_duplicates=dropped,
        duplicate_policy='all identical decoded RGB copies removed across selected samples',
        ood_groups=sorted(k[4:] for k in sources if k.startswith('ood:')),
        evaluation_variant='selected, pixel-deduplicated samples; report realized counts',
        ood_definition='outside declared ID class task, not necessarily unseen in pretraining')))


def prepare_builtin(root, output, id_dataset='cifar10', ood_datasets=('cifar100','svhn'),
                    counts=(500,100,100,100,100), seed=7, test_limit=None, download=False):
    allowed={'cifar10','cifar100','svhn'}
    if id_dataset not in allowed or not set(ood_datasets)<=allowed:
        raise ValueError('Built-in datasets are cifar10, cifar100, svhn')
    if id_dataset in ood_datasets or len(set(ood_datasets))!=len(ood_datasets):
        raise ValueError('OOD datasets must be distinct from ID and each other')
    root=str(Path(root).resolve())
    sources={key:dict(name=id_dataset,root=root,split=split)
             for key,split in (('train','train'),('id_test','test'))}
    sources.update({f'ood:{name}':dict(name=name,root=root,split='test') for name in ood_datasets})
    return build_manifest(sources,output,counts,seed,test_limit,download)


def prepare_folders(id_train,id_test,ood_folders,output,counts=(500,100,100,100,100),
                    seed=7,test_limit=None):
    sources={key:dict(name='imagefolder',root=str(Path(path).resolve()),split='files')
             for key,path in (('train',id_train),('id_test',id_test))}
    for spec in ood_folders:
        name,sep,path=spec.partition('=')
        if not sep or not name or f'ood:{name}' in sources:
            raise ValueError('OOD folders must be uniquely named NAME=/path entries')
        sources[f'ood:{name}']=dict(name='imagefolder',root=str(Path(path).resolve()),split='files')
    return build_manifest(sources,output,counts,seed,test_limit)
