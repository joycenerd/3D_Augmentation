import os
import random
import numpy as np 
import open3d as o3d

data_root = "/eva_data_6/datasets_raw/ModelNet40_auto_aligned_128/all/all_trian/"
instance_dirs = sorted(os.listdir(data_root))
cate_list = []
for instance_dir in instance_dirs:
    cate_list.append(instance_dir.split("_")[0])

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

def save_point_cloud(root_path, output_dir):
    # Note 2055, 3701, 3793 有問題
    for i in range(8500, 9843):
        random_view = random.sample(range(50), 10)
        cate = cate_list[i]
        points = []
        for view in random_view:
            content, _ = read_pcd(root_path, str(i), str(view))
            points.extend(content)
        
        if len(points) >= 1024:
            output_pcd_dir = os.path.join(output_dir, cate)
            os.makedirs(output_pcd_dir, exist_ok=True)
            write_pcd(np.array(points), output_pcd_dir + '/' + "%06d.pcd" %(int(i)))
            print("%06d" %(int(i)), np.array(points).shape, cate)

if __name__ == "__main__":
    
    root_path = "/home/zchin/augmentation_output/SRN/test/real_0.5"
    output_dir = "/home/zchin/augmentation_output/3D_points/real_0.5"
    
    save_point_cloud(root_path, output_dir)
