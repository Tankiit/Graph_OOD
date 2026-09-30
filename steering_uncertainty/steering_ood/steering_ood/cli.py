"""Shared OOD steering for text and vision: prepare, encode, train, evaluate, cross."""
import argparse
import json
from pathlib import Path
from .core import jsonable
from .experiment import Config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare',help='Partition official CLINC data_full.json; excludes OOS train/val')
    a.add_argument('--source',required=True); a.add_argument('--output',required=True)
    a.add_argument('--seed',type=int,default=20260929)
    a=sub.add_parser('encode',help='Encode a prepared text split with a frozen encoder')
    a.add_argument('--splits',required=True); a.add_argument('--output',required=True)
    a.add_argument('--model',default='sentence-transformers/all-mpnet-base-v2')
    a.add_argument('--revision'); a.add_argument('--batch-size',type=int,default=64)
    a.add_argument('--device',default='cpu')
    a=sub.add_parser('prepare-vision',help='Prepare built-in CIFAR10/CIFAR100/SVHN image splits')
    a.add_argument('--root',required=True); a.add_argument('--output',required=True)
    a.add_argument('--id-dataset',choices=['cifar10','cifar100','svhn'],default='cifar10')
    a.add_argument('--ood-datasets',nargs='+',default=['cifar100','svhn'])
    a.add_argument('--counts',type=int,nargs=5,default=[500,100,100,100,100],
                   metavar=('HEAD','REFERENCE','DIRECTION','CALIBRATION','PROBE'))
    a.add_argument('--seed',type=int,default=7); a.add_argument('--test-limit',type=int)
    a.add_argument('--download',action='store_true')
    a=sub.add_parser('prepare-imagefolder',help='Prepare custom image folders, one subfolder per class')
    a.add_argument('--id-train',required=True); a.add_argument('--id-test',required=True)
    a.add_argument('--ood-folders',nargs='+',required=True,help='NAME=/path entries')
    a.add_argument('--output',required=True)
    a.add_argument('--counts',type=int,nargs=5,default=[500,100,100,100,100])
    a.add_argument('--seed',type=int,default=7); a.add_argument('--test-limit',type=int)
    a=sub.add_parser('encode-vision',help='Export pooled features with checkpoint-specific preprocessing')
    a.add_argument('--splits',required=True); a.add_argument('--output',required=True)
    a.add_argument('--backend',choices=['torchvision','timm'],default='torchvision')
    a.add_argument('--model',default='resnet18'); a.add_argument('--weights',default='DEFAULT')
    a.add_argument('--batch-size',type=int,default=64); a.add_argument('--device',default='cpu')
    a.add_argument('--seed',type=int,default=7)
    a=sub.add_parser('import-features',help='Validate existing disjoint feature arrays and attach provenance')
    a.add_argument('--source',required=True); a.add_argument('--metadata',required=True)
    a.add_argument('--output',required=True)
    a=sub.add_parser('project',help='Optional PCA fitted on head-training features only')
    a.add_argument('--cache',required=True); a.add_argument('--output',required=True)
    a.add_argument('--components',type=int,default=64)
    a=sub.add_parser('fixture',help='Generate synthetic features for integration checks (not NLP results)')
    a.add_argument('--output',required=True); a.add_argument('--seed',type=int,default=7)
    a=sub.add_parser('train-head',help='Train an ID class head on the head split only')
    a.add_argument('--cache',required=True); a.add_argument('--output',required=True)
    a.add_argument('--seed',type=int,default=7); a.add_argument('--epochs',type=int,default=30)
    a.add_argument('--lr',type=float,default=.01); a.add_argument('--batch-size',type=int,default=128)
    a=sub.add_parser('e0',help='Run analytical validity checks')
    a.add_argument('--output',default='e0.json'); a.add_argument('--include-pytorch',action='store_true')
    for name in ('evaluate','crossed','synthetic'):
        a=sub.add_parser(name)
        if name!='synthetic':
            a.add_argument('--cache',required=True); a.add_argument('--head')
        else:
            a.add_argument('--n-reference',type=int,default=64)
            a.add_argument('--n-direction',type=int,default=64)
            a.add_argument('--dim',type=int,default=6)
        a.add_argument('--output',required=True)
        a.add_argument('--detectors',nargs='+',default=['gaussian','knn'] if name=='synthetic' else ['knn','mahalanobis','energy','msp'])
        a.add_argument('--backend',choices=['pytorch','sklearn'],default='pytorch')
        a.add_argument('--seed',type=int,default=7); a.add_argument('--tau',type=float,default=.05)
        a.add_argument('--k',type=int,default=5)
        a.add_argument('--temperatures',type=float,nargs='+',default=[1.])
        a.add_argument('--references',type=int,default=10); a.add_argument('--directions',type=int,default=10)
        a.add_argument('--probes',type=int,default=128); a.add_argument('--horizon',type=float,default=5.)
        a.add_argument('--steps',type=int,default=101); a.add_argument('--refine',type=int,default=15)
        a.add_argument('--direction-kind',choices=['pca','random','oracle','sphere_pca','sphere_random','origin','radial'],default='pca')
        a.add_argument('--calibration-draws',type=int,default=0)
    args=p.parse_args(); result=None
    if args.command=='prepare':
        from .data import prepare_clinc
        prepare_clinc(args.source,args.output,args.seed)
    elif args.command=='encode':
        from .data import encode
        encode(args.splits,args.output,args.model,args.revision,args.batch_size,args.device)
    elif args.command=='prepare-vision':
        from .vision_data import prepare_builtin
        prepare_builtin(args.root,args.output,args.id_dataset,args.ood_datasets,
                        args.counts,args.seed,args.test_limit,args.download)
    elif args.command=='prepare-imagefolder':
        from .vision_data import prepare_folders
        prepare_folders(args.id_train,args.id_test,args.ood_folders,args.output,
                        args.counts,args.seed,args.test_limit)
    elif args.command=='encode-vision':
        from .vision import encode_vision
        encode_vision(args.splits,args.output,args.backend,args.model,args.weights,
                      args.batch_size,args.device,args.seed)
    elif args.command=='import-features':
        from .data import import_features
        import_features(args.source,args.metadata,args.output)
    elif args.command=='project':
        from .data import project_cache
        project_cache(args.cache,args.output,args.components)
    elif args.command=='fixture':
        from .data import synthetic_cache
        synthetic_cache(args.output,args.seed)
    elif args.command=='train-head':
        from .head import train_head
        train_head(args.cache,args.output,args.seed,args.epochs,args.lr,args.batch_size)
    elif args.command=='e0':
        from .audit import run_e0
        result=run_e0(args.output,args.include_pytorch)
    else:
        from .experiment import crossed,evaluate,synthetic_crossed
        fields=Config.__dataclass_fields__
        cfg=Config(**{k:getattr(args,k) for k in fields})
        head=None
        if getattr(args,'head',None):
            from .head import load_head
            head=load_head(args.head,args.cache)
        if args.command=='synthetic':
            result=synthetic_crossed(args.output,cfg,args.n_reference,args.n_direction,args.dim)
        else:
            result=(crossed if args.command=='crossed' else evaluate)(args.cache,args.output,cfg,head)
    print(json.dumps(jsonable(result if result is not None else {'saved':args.output}),indent=2))
