import os
import struct
import collections
import imageio
import numpy as np
from shutil import copyfile


root = "/work/eva0856121/datasets/NeRF/nerf_llff_data/"
dis = "/work/eva0856121/datasets/srn_data/llff_train/"

intrinsic = {
    "fern" : [3260.5263328805895, 2016.0, 1512.0, 0.019750463822896944],
    "flower" : [3575.0586059510074, 2016.0, 1512.0, 0.025134574603909228],
    "fortress" : [3371.3189700388566, 2016.0, 1512.0, 0.007315123920083764],
    "horns" :[3368.8237176028883, 2016.0, 1512.0, 0.012300143763124053],
    "leaves" : [3428.4755177313386, 2016.0, 1512.0, 0.03738835450733112],
    "orchids" : [3124.62276683125, 2016.0, 1512.0, 0.03324880811508114],
    "room" : [3070.63827088164, 2016.0, 1512.0, 0.009615941992791891],
    "trex" : [3329.8699571672205, 2016.0, 1512.0, 0.01265264013010501]
}
'''
intrinsic = {
    "Truck": [1.15882652e+03,  9.60000000e+02,  5.40000000e+02, -2.91480825e-02]
}
'''
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


scenes = os.listdir(root)
# scenes = ["Truck"]
for scene in scenes:
    print("Now we are create ", scene)
    
    # Create intrinsic file
    with open(os.path.join(dis, scene, "intrinsics.txt"), "w") as fp:
        for element in intrinsic[scene]:
            print(element, end=" ", file=fp)
        print("", file=fp)
        print("0. 0. 0.", file=fp)
        print("1.", file=fp)
        img_path = os.path.join(root, scene, "images")
        images = sorted(os.listdir(img_path))
        img = imageio.imread(os.path.join(img_path, images[0]))[:, :, :3]
        print("{} {}".format(img.shape[1], img.shape[0]), file=fp)
        
    # Create poses dir
    pose_dir = os.path.join(dis, scene, "pose")
    os.makedirs(pose_dir, exist_ok=True)

    imagesfile = os.path.join(root, scene, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
     
    names = [imdata[k].name for k in imdata]
    print(names)
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        m = m.reshape(-1)
        with open(os.path.join(pose_dir, names[k-1].replace(".jpg", ".txt").replace(".JPG", ".txt").replace(".png", ".txt").replace(".PNG", ".txt")), "w") as fp:
            for element in m:
                print(element, end=" ", file=fp)
