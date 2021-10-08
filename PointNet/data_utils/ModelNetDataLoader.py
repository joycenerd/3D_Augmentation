import os
import warnings
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import sys
sys.path.append("/home/zchin/3D_Augmentation/data_utils")
from utils import *

warnings.filterwarnings('ignore')
root_dirs = {"ModelNet40": "/eva_data_6/datasets_ours/MN40",
             "ModelNet10": "/eva_data/psa/datasets/PointNet/ModelNet10_pcd",
             "ShapeNet": "/eva_data/psa/datasets/MSN_PointNet/ShapeNetCore.v1",
             "ShapeNet_all": "/eva_data/psa/datasets/MSN_PointNet/ShapeNetCore.v1_all_10000"}
obj_name_files = {"ModelNet40": "ModelNet40_names.txt", #"ModelNet40_ourDA10_names.txt", #"ModelNet40_names.txt",
                  "ModelNet10": "modelnet10_shape_names.txt",
                  "ShapeNet": "ShapeNetCore.v1_ID.txt",
                  "ShapeNet_all": "ShapeNetCore.v1_all_ID.txt"}
# root_dir = "/eva_data/psa/datasets/PointNet/ModelNet40_pcd_SampleNet/64_zorder_keep50"

def rotate_point_cloud(pcd):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    # Create rotation matrix
    angle_x = np.random.uniform() * 2 * np.pi
    # angle_y = np.random.uniform() * 2 * np.pi
    angle_y = np.random.choice([-1, -0.5, 0.5, 1], 1)[0] * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])

    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])

    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    # Rotate
    # rotated_pcd = np.dot(rotated_pcd, Rz)
    # R = np.dot(Rz, np.dot(Ry, Rx))
    R = Ry
    rotated_pcd = np.dot(pcd, R)

    return rotated_pcd.astype(np.float32)

class ModelNetDataLoader(Dataset):
    def __init__(self,  npoint=1024, split="train", sparsify_mode="PN", dataset_mode="ModelNet40", zorder_mode="keep", cache_size=15000):
        self.npoints = npoint
        self.sparsify_mode = "random"  #sparsify_mode
        self.zorder_mode = zorder_mode

        assert (dataset_mode == "ModelNet40" or dataset_mode == "ModelNet10" or dataset_mode == "ShapeNet" or dataset_mode == "ShapeNet_all"), "PLZ verify dataset_mode should be [ModelNet40, ModelNet10, ShapeNet, ShapeNet_all]"
        root_dir = root_dirs[dataset_mode]
        obj_name_file = obj_name_files[dataset_mode]

        # Load class name file and Create target table
        self.class_file = os.path.join(root_dir, obj_name_file)
        self.class_names = [line.rstrip() for line in open(self.class_file)]
        self.target_table = dict(zip(self.class_names, range(len(self.class_names))))
        print(self.target_table)

        # Create datapath -> (shape_name, shape_pcd_file_pat)
        assert (split == "train" or split == "test"), "PLZ verify split should be [train, test]"
        self.datapath = [] # list of (shape_name, shape_pcd_file_path) tuple
        for class_name in self.class_names:
            file_dir = os.path.join(root_dir, class_name, split)
            # print(file_dir)
            if os.path.exists(file_dir):
                filenames = os.listdir(file_dir)
                for filename in filenames:
                    # if "aug" not in filename:
                    file_path = os.path.join(file_dir, filename)
                    self.datapath.append((class_name, file_path))
        
        print("The size of %s data is %d" % (split, len(self.datapath)))

        # Record loaded data
        self.cache_size = cache_size  # How many data points to cache in memory
        self.cache = {}  # From index to (points, target) tuple

    def __getitem__(self, index):
        if index in self.cache:
            # Use duplicated data in cache memory
            points, target = self.cache[index]
            print("Using cache !!!")
        else:
            classname_target_pair = self.datapath[index]

            # Get target
            target = self.target_table[classname_target_pair[0]]
            target = np.array([target]).astype(np.int32)
            
            # Get points
            points = read_pcd(classname_target_pair[1])
            points = pcd_normalize(points) # Normalize points

            # Rotate augmentation
            if "aug_" in classname_target_pair[1]:
                points = rotate_point_cloud(points)

            # Various sparsify mode
            if self.sparsify_mode == "PN":
                points = points[:self.npoints, :]
            elif self.sparsify_mode == "random":
                # points = random_sample(points, self.npoints)
                points = resample_pcd(points, self.npoints)
            elif self.sparsify_mode == "fps":
                points = farthest_point_sample(points, self.npoints)
            elif "zorder" in self.sparsify_mode:
                z_values = get_z_values(points) # Get z values for all points
                points_zorder = points[np.argsort(z_values)] # Sort input points with z values

                if self.sparsify_mode == "zorder":
                    points = keep_zorder(points_zorder) if self.zorder_mode == "keep" else discard_zorder(points_zorder, self.npoints, len(points_zorder))
                elif self.sparsify_mode == "multizorder":
                    points = keep_multizorder(points_zorder) if self.zorder_mode == "keep" else discard_multizorder(points_zorder, self.npoints, len(points_zorder))
                else:
                    assert False, "You should choose [zorder] or [multizorder]"
            else:
                assert False, "PLZ verify sparsify mode is in [PN, random, fps, zorder, multizorder] or not"
            
            # Record loaded (points, target)
            # if len(self.cache) < self.cache_size:
            #     self.cache[index] = (points, target)
        
        return points, target

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader(split="train")
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    #for point,label in DataLoader:
        #print(point.shape)
        #print(label.shape)
