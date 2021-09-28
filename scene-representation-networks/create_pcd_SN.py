import os
import random
import numpy as np 
import open3d as o3d

SN2MN = {
	"bathtub": "bathtub",
	"bed": "bed",
	"bookshelf": "bookshelf",
	"chair": "chair",
	"lamp": "lamp",
	"sofa": "sofa",
	"table": "table",
	"display": "monitor",
	"cabinet": "night_stand",
	"flowerpot": "plant"
}

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

def save_point_cloud(cate_path, cate, output_dir):

    id_list = sorted(list(set([path.split('_')[0] for path in os.listdir(cate_path) if ".npy" in path])))
    for i in id_list:
    # for i in range(1089, 1137):
        random_view = random.sample(range(50), 10)
        points = []
        for view in random_view:
            content, _ = read_pcd(cate_path, str(i), str(view))
            points.extend(content)
        
        if len(points) >= 1024:
            output_pcd_dir = os.path.join(output_dir, SN2MN[cate])
            os.makedirs(output_pcd_dir, exist_ok=True)
            write_pcd(np.array(points), output_pcd_dir + '/' + "%06d.pcd" %(int(i)))
            print("%06d" %(int(i)), np.array(points).shape, cate)

if __name__ == "__main__":
    root_path = "./3D_points/shapenet.v2_DA10_VI_inter_more/7_3"
    # categories = ["bathtub", "bed", "bookshelf", "cabinet", "chair", "lamp", "display", "flowerpot", "sofa", "table"]
    categories = ["bookshelf"]
    output_dir = "/work/eva0856121/Augmentation/datasets/SN_class_DA10_VI_inter_10views/"
    for cate in categories:
        cate_path = os.path.join(root_path, cate+"_train")
        save_point_cloud(cate_path, cate, output_dir)
