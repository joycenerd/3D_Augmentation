import argparse
import os
from pathlib import Path
import math

import numpy as np


parser=argparse.ArgumentParser()
parser.add_argument('--ratio',type=float,default=0.5,help='real data ratio')
parser.add_argument('--data-root',type=str,default='/eva_data_6/datasets_raw/ModelNet40_auto_aligned',help='complete training data path')
parser.add_argument('--out-dir',type=str,default='/home/zchin/augmentation_output/train_ratio_data')
args=parser.parse_args()

args.out_dir=f'{args.out_dir}/{args.ratio}'
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)
f_path=f'{args.out_dir}/train_data_path.txt'
f=open(f_path,'w',encoding='utf-8')

all_category=os.listdir(args.data_root)
for category in all_category:
    if category=='modelnet40_auto_aligned.tar':
        continue
    train_dir=Path(args.data_root).joinpath(category,'train')
    entries=os.listdir(train_dir)
    n=math.ceil(len(entries)*args.ratio)
    print(f'{category}: {n}')
    idx=np.random.randint(low=0,high=len(entries),size=n)
    for i in idx:
        entry='/eva_data_6/datasets_raw/ModelNet40_auto_aligned_128/all/all_train/'+entries[i]
        f.write(entry)
        f.write('\n')
f.close()



