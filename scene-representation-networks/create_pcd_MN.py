import os
import random
import numpy as np 
import open3d as o3d
import re
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--ratio',type=float, required=True, help='real data ratio')
parser.add_argument('--iter',type=int,required=True,help='SRN training iteration')
parser.add_argument('--class_name',type=str,required=True,help='The class want to train')
args=parser.parse_args()

# data_root = "/eva_data_6/datasets_raw/ModelNet40_auto_aligned_128/all/all_train/"
# instance_dirs = sorted(os.listdir(data_root))
# cate_list = []
# for instance_dir in instance_dirs:
#     cate_list.append(instance_dir.split("_")[0])

def pred_2_rgb(img):
    img += 1.
    img /= 2.
    img *= 2**8 - 1
    img = img.round().clip(0, 2**8-1)
    return img

def read_pcd(cate_path, instance_id, pose_id):
    content = np.load(cate_path + '/' + instance_id + "_" + str(pose_id).zfill(6)+ "_points.npy")
    content = content[0]
    
    color_rgb = np.load(cate_path + '/' + instance_id + "_" + str(pose_id).zfill(6)+ "_rgb.npy")
    color_rgb = color_rgb[0]

    img = pred_2_rgb(color_rgb)
    filt = np.where(np.average(img, axis=1) <= 230)
    
    return content[filt], img[filt]    

def write_pcd(point, output_pcd_path):
    # Convert numpy array to pcd format
    pcd = o3d.geometry.PointCloud()
    try:
        point = np.random.choice(point, 16384, reaplace=False)
    except:
        point = point
    pcd.points = o3d.utility.Vector3dVector(point)

    # Output pcd file
    o3d.io.write_point_cloud(output_pcd_path, pcd)

def load_test_dirs(choosen_path):
    f = open(choosen_path,'r', encoding='utf-8')
    choosen_dirs = []
    for line in f:
        choosen_dirs.append(line.strip())
    return choosen_dirs

def save_point_cloud(root_path, output_dir, choosen_path):
    # Note 2055, 3701, 3793 有問題
    test_dirs = load_test_dirs(choosen_path)
    for i in range(len(test_dirs)):
        random_view = random.sample(range(50), 10)
        cate = test_dirs[i].split("/")[-1][:-5]
        points = []
        for view in random_view:
            content, _ = read_pcd(root_path, str(i), str(view))
            points.extend(content)
        if len(points) > 0:
            output_pcd_dir = os.path.join(output_dir, cate)
            os.makedirs(output_pcd_dir, exist_ok=True)
            write_pcd(np.array(points), output_pcd_dir + '/' + "%06d.pcd" %(int(i)))
            print("%06d" %(int(i)), np.array(points).shape, cate)

if __name__ == "__main__":
    
    root_path = f"/eva_data_0/augment_output_single_version/SRN/test/{args.class_name}/real_{args.ratio}/3D_info"
    output_dir = f"/eva_data_0/augment_output_single_version/3D_points/iter_{args.iter}/{args.class_name}/real_{args.ratio}"
    choosen_path = f"/eva_data_0/augment_output_single_version/ratio_data/{args.class_name}/{args.ratio}/test_data_path.txt"
    save_point_cloud(root_path, output_dir, choosen_path)
