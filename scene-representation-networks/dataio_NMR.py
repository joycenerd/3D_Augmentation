import os
import torch
import numpy as np
from glob import glob
from random import sample
import data_util
import util

name2id = {
    "airplane" : "02691156",
    "bench" : "02828884",
    "cabinet" : "02933112",
    "car" : "02958343",
    "chair" : "03001627",
    "display" : "03211117",
    "lamp" : "03636649",
    "speaker" : "03691459",
}

def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=-1):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir

        color_dir = os.path.join(instance_dir, "image")
        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        
        cam_path = os.path.join(instance_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        
        pose_key =  "world_mat_"
        intrinsic_key = "camera_mat_"
        self.poses = []
        self.intrinsics = []
        for i in range(len(self.color_paths)):
            self.poses.append(all_cam[pose_key+str(i)])
            self.intrinsics.append(all_cam[intrinsic_key+str(i)])

        self.intrin = self.intrinsics[0]
        self.intrin[0][0] = self.intrin[0][0] * self.img_sidelength
        self.intrin[1][1] = self.intrin[1][1] * self.img_sidelength
        # self.intrin[0][0] = self.img_sidelength 
        # self.intrin[1][1] = self.img_sidelength 
        self.intrin[0][2] = self.img_sidelength / 2
        self.intrin[1][2] = self.img_sidelength / 2 
            
        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.poses = pick(self.poses, specific_observation_idcs)
            self.intrinsics = pick(self.intrinsics, specific_observation_idcs)
            
        elif num_images != -1:
            # first num_images
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            # Random one image
            # idcs = sample(list(np.arange(50)), num_images)
            self.color_paths = pick(self.color_paths, idcs)
            self.poses = pick(self.poses, idcs)
            self.intrinsics = pick(self.intrinsics, idcs)
            
            
    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        
        rgb = data_util.load_rgb(self.color_paths[idx], sidelength=self.img_sidelength)
        rgb = rgb.reshape(3, -1).transpose(1, 0)

        intrin = torch.Tensor(self.intrin).float()
        
        # For test
        # input_img = data_util.load_rgb(self.color_paths[idx if "test" not in self.instance_dir else 64], sidelength=self.img_sidelength)
        input_img = data_util.load_rgb(self.color_paths[idx], sidelength=self.img_sidelength)
        
        pose = self.poses[idx]

        params = np.array([0])

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "input_img": torch.from_numpy(input_img).float(),
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
            "rgb": torch.from_numpy(rgb).float(),
            "pose": torch.from_numpy(pose).float(),
            "uv": uv,
            "param": torch.from_numpy(params).float(),
            "intrinsics": intrin,
        }
        return sample


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 root_dir,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 samples_per_instance=2,
                 train_class= "all"):

        self.samples_per_instance = samples_per_instance
        # Read all the instances from datasets        
        if train_class == "all":
            self.instance_dirs = []
            for class_name in name2id.keys():
                single_class_dirs = sorted(glob(os.path.join(root_dir, name2id[class_name], "*/")))
                if max_num_instances != -1:
                    single_class_dirs = single_class_dirs[:max_num_instances]
                self.instance_dirs.extend(single_class_dirs)
        else: 
            self.instance_dirs = sorted(glob(os.path.join(root_dir, name2id[train_class], "*/")))
            if max_num_instances != -1:
                self.instance_dirs = self.instance_dirs[:max_num_instances]
        
        assert (len(self.instance_dirs) != 0), "No objects in the data directory"
        
        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance)
                              for idx, dir in enumerate(self.instance_dirs)]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        for _ in range(self.samples_per_instance - 1):
            observations.append(self.all_instances[obj_idx][np.random.randint(len(self.all_instances[obj_idx]))])

        ground_truth = [{'rgb':ray_bundle['rgb']} for ray_bundle in observations]

        return observations, ground_truth


if __name__ == "__main__":
    dataset = SceneClassDataset(root_dir = "/eva_data/psa/datasets/NMR_Dataset",
                                max_num_instances=200,
                                max_observations_per_instance=24,
                                img_sidelength=64,
                                # specific_observation_idcs=[64,194],
                                samples_per_instance=50,
                                train_class="all")
    import pdb; pdb.set_trace()
