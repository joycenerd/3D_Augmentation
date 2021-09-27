# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("/work/eva0856121/Augmentation/code/data_utils")
from data_utils.utils import *
from PointNet import provider

# mpl.use("TkAgg")

parser = argparse.ArgumentParser()
parser.add_argument("--path_1", type=str, default="/eva_data/psa/source_code/PCN/demo_data/car.pcd", help="1st input data path.")
parser.add_argument("--path_2", type=str, default="/eva_data/psa/source_code/PCN/demo_data/airplane.pcd", help="2nd input data path.")
parser.add_argument("--path_3", type=str, default="/eva_data/psa/source_code/PCN/demo_data/chair.pcd", help="3rd input data path.")
parser.add_argument("--path_4", type=str, default="/eva_data/psa/source_code/PCN/demo_data/lamp.pcd", help="4th input data path.")
parser.add_argument("--dir_path", type=str, default=None, help="The directory path of input data.")
parser.add_argument("--dir_path_sample", type=str, default=None, help="The directory path of input data.")
parser.add_argument("--certain_obj", type=str, default=None, help="The certain object in directory path you want to visualize.")
args = parser.parse_args()

def alignment(pcd):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    # Create rotation matrix
    angle_x = np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    # Rotate
    rotated_pcd = np.dot(pcd, Rx)

    return rotated_pcd

def my_normalize(pcd):
    '''
    # Normalize pcd based the axis which has the maximum range
    pcd_x_range = np.max(pcd[:, 0]) - np.min(pcd[:, 0])
    pcd_y_range = np.max(pcd[:, 1]) - np.min(pcd[:, 1])
    pcd_z_range = np.max(pcd[:, 2]) - np.min(pcd[:, 2])
    max_range_idx = np.argmax([pcd_x_range, pcd_y_range, pcd_z_range])
    pcd = ((pcd - np.min(pcd[:, max_range_idx])) / (np.max(pcd[:, max_range_idx]) - np.min(pcd[:, max_range_idx]))) - 0.5
    print("Normalized pcd: {}\n{}".format(pcd.shape, pcd[:5]))
    '''

    '''
    # Normalize pcd along all axis
    pcd[:, 0] = ((pcd[:, 0] - np.min(pcd[:, 0])) / (np.max(pcd[:, 0]) - np.min(pcd[:, 0]))) - 0.5
    pcd[:, 1] = ((pcd[:, 1] - np.min(pcd[:, 1])) / (np.max(pcd[:, 1]) - np.min(pcd[:, 1]))) - 0.5
    pcd[:, 2] = ((pcd[:, 2] - np.min(pcd[:, 2])) / (np.max(pcd[:, 2]) - np.min(pcd[:, 2]))) - 0.5
    print("Normalized pcd: {}\n{}".format(pcd.shape, pcd[:5]))
    '''

    '''
    # Crop pcd in range [-0.5, 0.5]
    pcd = pcd[abs(pcd[:, 0]) <= 0.5]
    pcd = pcd[abs(pcd[:, 1]) <= 0.5]
    pcd = pcd[abs(pcd[:, 2]) <= 0.5]
    print("Normalized pcd: {}\n{}".format(pcd.shape, pcd[:5]))
    '''
    
    return pcd

def plot_pcd(ID, fig, pcd, title, split="original"):
    if split == "original":
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir="y", c=[1, 0, 0], s=0.5, cmap="Reds", vmin=-1, vmax=0.5)
    else:
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir="y", c=[0, 0, 1], s=5, cmap="Reds", vmin=-1, vmax=0.5)
    # ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=np.arange(0, len(pcd)), zdir="y", s=0.5)
    # ax.set_axis_off()
    limit = 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_title("\n" + title, fontsize=10)

if __name__ == "__main__":
    # Load pcd data
    pcds = {"original": [], "sampled": []}
    filepaths = {"original": [], "sampled": []}
    if args.dir_path is None:
        for path in [args.path_1, args.path_2, args.path_3, args.path_4]:
            pcd = read_pcd(path)
            # pcd = pcd_normalize(pcd)
            # print(pcd.shape)
            # pcd_z_values = get_z_values(pcd)
            # pcd = pcd[np.argsort(pcd_z_values)]
            # pcd = discard_multizorder(pcd, 5000, len(pcd))
            # print(pcd.shape)
            pcds["original"].append(pcd)
            filepaths["original"].append(path)
            '''
            pcd = read_pcd(path)
            pcd = pcd_normalize(pcd)
            pcd = get_zorder_sequence(pcd)
            pcd_sample_0 = discard_zorder(pcd, 1024, len(pcd))
            pcd_sample = get_zorder_sequence(random_sample(pcd, 7500))
            pcd_sample_1 = discard_zorder(pcd_sample, 1024, len(pcd_sample))
            pcd_sample = get_zorder_sequence(random_sample(pcd, 5000))
            pcd_sample_2 = discard_zorder(pcd_sample, 1024, len(pcd_sample))
            pcd_sample = get_zorder_sequence(random_sample(pcd, 2500))
            pcd_sample_3 = discard_zorder(pcd_sample, 1024, len(pcd_sample))
            pcds["original"].append(pcd_sample_0)
            filepaths["original"].append("Zorder sample from 10,000 points")
            pcds["original"].append(pcd_sample_1)
            filepaths["original"].append("Zorder sample from 7,000 points")
            pcds["original"].append(pcd_sample_2)
            filepaths["original"].append("Zorder sample from 5,000 points")
            pcds["original"].append(pcd_sample_3)
            filepaths["original"].append("Zorder sample from 2,000 points")
            break
            '''
    else:
        filenames = sorted(os.listdir(args.dir_path))
        for filename in filenames:
            if (args.certain_obj is None) or ((args.certain_obj is not None) and (args.certain_obj in filename)):
                filepath = os.path.join(args.dir_path, filename)
                # filepath_sample = os.path.join(args.dir_path_sample, filename)
                pcd = read_pcd(filepath)
                pcd = resample_pcd(pcd)
                pcd = pcd_normalize(pcd)
                
                # Do something like augmentation
                # pcd[np.newaxis, :, 0:3] = provider.random_scale_point_cloud(pcd[np.newaxis,:, 0:3])
                # pcd[np.newaxis, :, 0:3] = provider.jitter_point_cloud(pcd[np.newaxis, :, 0:3])
                # pcd[np.newaxis, :, 0:3] = provider.rotate_point_cloud_z(pcd[np.newaxis, :, 0:3])
                
                # pcd = alignment(pcd)
                # print(pcd.shape)
                # pcd = resample_pcd(pcd)
                # pcd_z_values = get_z_values(pcd)
                # pcd = pcd[np.argsort(pcd_z_values)]
                # pcd = discard_fps_multizorder(pcd, 1024, len(pcd))
                pcds["original"].append(pcd)
                filepaths["original"].append(filepath)

                # pcds_sample.append(pcd_sample)
                # filepaths_sample.append(filepath_sample)
    
    # Visualization
    for idx in range(0, len(pcds["original"]), 4):
        print("Index: {} -> {}".format(idx+1, filepaths["original"][idx]))
        print("Index: {} -> {}".format(idx+2, filepaths["original"][idx+1]))
        print("Index: {} -> {}".format(idx+3, filepaths["original"][idx+2]))
        print("Index: {} -> {}".format(idx+4, filepaths["original"][idx+3]))

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(221, projection="3d")
        plot_pcd(221, fig, pcds["original"][idx], filepaths["original"][idx].split("/")[-1])
        # plot_pcd(221, fig, pcds["original"][idx+1], filepaths["original"][idx].split("/")[-1], split="sample")
        ax = fig.add_subplot(222, projection="3d")
        plot_pcd(222, fig, pcds["original"][idx+1], filepaths["original"][idx+1].split("/")[-1])
        # plot_pcd(222, fig, pcds_sample[idx+1], filepaths[idx+1].split("/")[-1], split="sample")
        ax = fig.add_subplot(223, projection="3d")
        plot_pcd(223, fig, pcds["original"][idx+2], filepaths["original"][idx+2].split("/")[-1])
        # plot_pcd(223, fig, pcds_sample[idx+2], filepaths[idx+2].split("/")[-1], split="sample")
        ax = fig.add_subplot(224, projection="3d")
        plot_pcd(224, fig, pcds["original"][idx+3], filepaths["original"][idx+3].split("/")[-1])
        # plot_pcd(224, fig, pcds_sample[idx+3], filepaths[idx+3].split("/")[-1], split="sample")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
        plt.show()
