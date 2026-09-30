"""Download-free image -> real ResNet -> Skorch -> OOD -> crossed integration.

Uses generated image fixtures and random ResNet weights by default. These outputs
are software checks, never a vision benchmark. --pretrained loads ImageNet weights.
Run from the extracted project root after installing .[vision].
"""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image

from steering_ood.vision_data import prepare_folders
from steering_ood.vision import encode_vision
from steering_ood.data import project_cache
from steering_ood.head import train_head,load_head
from steering_ood.experiment import Config,evaluate,crossed
from steering_ood.core import new_output,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='runs/vision-smoke')
    p.add_argument('--pretrained',action='store_true')
    args=p.parse_args();out=new_output(args.output)
    rng=np.random.default_rng(83)
    for source,n in [('train',24),('test',4),('ood',4)]:
        for c in range(2):
            folder=out/'images'/source/f'class{c}';folder.mkdir(parents=True)
            for i in range(n):
                pixels=rng.integers(0,256,(32,32,3),dtype='uint8')
                Image.fromarray(pixels).save(folder/f'{i:03d}.png')
    prepare_folders(out/'images/train',out/'images/test',[f'fixture={out}/images/ood'],
                    out/'splits.json',counts=(3,3,3,12,3))
    encode_vision(out/'splits.json',out/'resnet.npz',weights='DEFAULT' if args.pretrained else 'none',batch_size=8)
    project_cache(out/'resnet.npz',out/'projected.npz',components=4)
    train_head(out/'projected.npz',out/'head',epochs=3)
    head=load_head(out/'head',out/'projected.npz')
    cfg=Config(references=2,directions=2,probes=2,steps=11,refine=3,tau=.2)
    evaluate(out/'projected.npz',out/'static',cfg,head)
    crossed(out/'projected.npz',out/'crossed',cfg,head)
    write_json(out/'CHECK_ONLY.json',dict(status='completed',modality='vision',
        dataset='generated image fixture',encoder='torchvision resnet18',
        pretrained=args.pretrained,benchmark_result=False,
        note='20% calibration target only for the tiny integration fixture; paper default is 5%'))
    print(json.dumps({'saved':str(out),'benchmark_result':False}))


if __name__=='__main__':
    main()
