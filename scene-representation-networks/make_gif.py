import os
import imageio

if __name__ == "__main__":
    root_dir = "/work/eva0856121/code/scene-representation-networks/logs/one_shot_cars_new_iter175000/regression_200_randomview/renderings/000007"
    images = []
    filenames = sorted(os.listdir(root_dir))

    for filename in filenames:
        images.append(imageio.imread(os.path.join(root_dir, filename)))
    
    imageio.mimsave(os.path.join("./", root_dir.split("/")[-3] + "_" + root_dir.split("/")[-1] + ".gif"), images)