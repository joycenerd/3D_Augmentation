# 3D Point Cloud Data Augmentation via Scene Representation Network
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Teammate
* Zhi-Yi Chin: joycenerd.cs09@nycu.edu.tw
* Chieh-Ming Jiang: nax1016.cs10@nycu.edu.tw

## Introduction
We design a 3D point cloud augmentation based on a novel view synthesis method, [SRN](https://arxiv.org/pdf/1906.01618.pdf). The 3D point cloud is a set of data points in 3D space, consisting of coordinates in world space and RGB color information. However, the precision is limited if there are insufficient points to describe the objects. Moreover, the amount of point cloud training samples is also limited, leading to the poor diversity of data. This motivates us to design a data augmentation method for the point cloud. As our base model, given extrinsic and intrinsic matrix, the SRN model can estimate the world coordinate and get the corresponding depth and RGB color information, which can easily be transferred to point cloud form. After the SRN model is trained, we can render new images of unseen views by changing the extrinsic matrix and consequently realize the point cloud augmentation. To verify whether the augmented point cloud is good enough to describe the 3D objects, we use [PointNet](https://arxiv.org/pdf/1612.00593.pdf) as our evaluation model to evaluate the quality of the augmentation.

[](./figure/pipeline.JPG)

## Getting the code
You can download a copy of all the files in this repository by cloning this repository:
```
git clone https://github.com/joycenerd/image-super-resolution.git
```

## Requirements
You need to have [Anaconda](https:www.anaconda.com/) or Miniconda already installed in your environment. To install requirements:
```
conda env create -f environment.yml
```

## Data

### Download raw data
* [ModelNet](https://modelnet.cs.princeton.edu/)
* [ShapeNet](https://shapenet.org/)

### Data conversion
* convert `.off` to `.obj`: 
    ```
    cd model-converter-python
    python convert.py \
--data-root <data_dir> \
--output-root <save_dir>
    ```
* Render 50 views (rgb, poses)

    Please refer to [here](shapenet_renderer/README.md) for detail.
* convert `.obj` to `.pcd`
    
    Please refer to [obj2pcd](https://github.com/jiangwei221/obj2pcd) for detail
* convert `.off` to `.ply`
    ```
    cd shapenet_renderer
    python off2ply.py --data-root <data_dir> --output-root <save_dir>
    ```
* convert `.ply` to `.pcd`
    You can use `open3d` api for conversion
    ```
    import open3d as o3d
    pcd = o3d.io.read_point_cloud("XXX.ply")
    o3d.io.write_point_cloud("XXX.ply", pcd)
    ```


## Train SRN
1. generate training and testing path
    ```
    cd scene-representation-networks
    python choose_train_test_data.py --ratio 0.7 --class_name airplane
    ```
2. Training
    ```
    cd scene-representation-networks
    python train.py --config_filepath train_configs/modelnet_all.yml --data_root train_data_path.txt --logging_root <save_dir> --ratio 0.7 --gpu 0 --train_class airplane
    ```
## Test SRN
1. Testing
    ```
    cd scene-representation-networks
    python test.py --config_filepath test_configs/modelnet_all.yml --data_root test_data_path.txt --logging_root <save_dir> --checkpoint <ckpt_path> --ratio 0.7 --gpu cuda:0 --train_class airplane --class_name airplane
    ```
2. Create point cloud points
    ```
    cd scene-representation-networks
    python create_pcd_MN.py --ratio 0.X --iter X --class_name XXX
    ```
## Train PointNet
```
cd PointNet
python train_cls.py --output_dir /eva_data_0/augment_output_single_version/PointNet/XXX/real_0.X/ --data-txt /eva_data_0/augment_output_single_version/ratio_data/XXX/0.X/train_data_path.txt --augment_data_dir /eva_data_0/augment_output_single_version/3D_points/XXX/real_0.X/ --gpu X
```

## Test PointNet
```
cd PointNet
python test_cls.py --output_dir <npy_files_dir> --gpu 0
```

## Results

[](figure/results.png)

## GitHub Acknowledge
We thank the authors of these repositories:
* [vsitzmann/scene-representation-networks](https://github.com/vsitzmann/scene-representation-networks)
* [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
* [tforgione/model-converter-python](https://github.com/tforgione/model-converter-python)
* [jiangwei221/obj2pcd](https://github.com/jiangwei221/obj2pcd)

## Citation
If you find our work useful in your project, please cite:

```bibtex
@misc{
    title = {3D Point Cloud Data Augmentation via Scene Representation Network},
    author = {Zhi-Yi Chin, Chieh-Ming Jiang},
    url = {https://github.com/joycenerd/3D_Augmentation},
    year = {2022}
}
```

## Contributing

If you'd like to contribute, or have any suggestions, you can contact us at [joycenerd.cs09@nycu.edu.tw](mailto:joycenerd.cs09@nycu.edu.tw) or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.


