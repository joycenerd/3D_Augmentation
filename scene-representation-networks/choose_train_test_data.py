import argparse
import os
from pathlib import Path
import math
import random
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, required=True , help='real data ratio')
parser.add_argument('--data-root', type=str, default='/eva_data_6/datasets_raw/ModelNet40_auto_aligned',help='complete training data path')
parser.add_argument('--out-dir', type=str, default='/home/zchin/augmentation_output/ratio_data')
args=parser.parse_args()

args.out_dir=f'{args.out_dir}/{args.ratio}'
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)


def load_train_data(root_dir):
    f = open(root_dir,'r', encoding='utf-8')
    choosen_dirs = []
    for line in f:
        choosen_dirs.append(line.strip())
    return choosen_dirs

def write_train_data():
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
        idx = random.sample(range(len(entries)), n)
        for i in idx:
            entry='/eva_data_6/datasets_raw/ModelNet40_auto_aligned_128/all/all_train/'+entries[i]
            f.write(entry)
            f.write('\n')
    f.close()

def write_test_data():
    choosen_dirs = load_train_data(f'{args.out_dir}/{"train_data_path.txt"}')
    remain_dirs = []
    all_category=os.listdir('/eva_data_6/datasets_raw/ModelNet40_auto_aligned')
    for category in all_category:
        if category=='modelnet40_auto_aligned.tar':
            continue
        train_dir=Path('/eva_data_6/datasets_raw/ModelNet40_auto_aligned').joinpath(category,'train')
        entries=os.listdir(train_dir)
        for i in range(len(entries)):
            entry='/eva_data_6/datasets_raw/ModelNet40_auto_aligned_128/all/all_train/'+entries[i]
            if entry not in choosen_dirs:
                remain_dirs.append(entry)
    remain_dirs.sort()
    print("Remain_dirs: ", len(remain_dirs), "choosen_dirs: " , len(choosen_dirs))


    path = "test_data_path.txt"
    f_path=f'{args.out_dir}/{path}'
    f=open(f_path,'w',encoding='utf-8')
    for remain_dir in remain_dirs:
        f.write(remain_dir)
        f.write('\n')
    f.close()


write_train_data()
write_test_data()

