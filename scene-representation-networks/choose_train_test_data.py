import argparse
import os
from pathlib import Path
import math
import random
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, required=True , help='real data ratio')
parser.add_argument('--data-root', type=str, default='/eva_data_eva_data_Augmentation/datasets_raw/ModelNet40_auto_aligned_128',help='complete training data path')
parser.add_argument('--out-dir', type=str, default='/eva_data_0/augment_output_single_version/ratio_data')
parser.add_argument('--class_name', type=str, required=True)
args=parser.parse_args()

args.out_dir=f'{args.out_dir}/{args.class_name}/{args.ratio}'
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)


def load_train_data(root_dir):
    f = open(root_dir,'r', encoding='utf-8')
    choosen_dirs = []
    for line in f:
        choosen_dirs.append(line.strip())
    return choosen_dirs

def write_train_data():
    f_path = f'{args.out_dir}/train_data_path.txt'
    f = open(f_path,'w',encoding='utf-8')
    train_dir = Path(args.data_root).joinpath(args.class_name, args.class_name + '_train')
    print(f'Splitting directory of {train_dir}')
    entries = sorted(os.listdir(train_dir))
    n = math.ceil(len(entries) * args.ratio)
    print(f'{args.class_name}: {n}/{len(entries)}')
    idx = random.sample(range(len(entries)), n)
    for i in idx:
        entry = f'{str(train_dir)}/{entries[i]}'
        f.write(entry)
        f.write('\n')
    f.close()

def write_test_data():
    choosen_dirs = load_train_data(f'{args.out_dir}/{"train_data_path.txt"}')
    remain_dirs = []
    train_dir = Path(args.data_root).joinpath(args.class_name, args.class_name + '_train')
    entries = os.listdir(train_dir)
    for i in range(len(entries)):
        entry = f'{str(train_dir)}/{entries[i]}'
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

