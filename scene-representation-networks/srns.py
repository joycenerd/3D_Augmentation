import glob
import os
import torch
import torch.nn as nn
import numpy as np

import torchvision
import util

import skimage.measure
from torch.nn import functional as F

from pytorch_prototyping import pytorch_prototyping
import custom_layers
import geometry
import hyperlayers

class AnalogyPred(nn.Module):
    def __init__(self, num_basis = 10, USE_EMBED=True, mode="regression"):
        super(AnalogyPred, self).__init__()
        input_dim = 256 if USE_EMBED == True else 512
        self.mode = mode
        if self.mode == "regression":
            self.linear1 = nn.Linear(input_dim, 256)
            self.linear2 = nn.Linear(256, 128)
            self.linear3 = nn.Linear(128, num_basis)
        else:
            self.linear1 = nn.Linear(in_features=input_dim, out_features=512)
            self.linear2 = nn.Linear(in_features=512, out_features=512)
            self.linear3 = nn.Linear(in_features=512, out_features=512)
    
        self.num_basis = num_basis
        self.BCELoss = nn.BCELoss()
        self.L1Loss = nn.L1Loss()
        self.cos_sim = nn.CosineSimilarity(dim=2)
    
    def get_bce_loss(self, pred_weights, gt_weights):
        return self.BCELoss(pred_weights, gt_weights)
    
    def get_l1_loss(self, pred_weights, gt_weights):
        return self.L1Loss(pred_weights, gt_weights)

    def forward(self, input, train_feat):
        batch_size = input.size()[0]

        input = torch.relu(self.linear1(input))
        input = torch.relu(self.linear2(input))
        input = self.linear3(input)
        
        if self.mode == "cos_sim":
            t_feat = torch.relu(self.linear1(train_feat.squeeze()))
            t_feat = torch.relu(self.linear2(t_feat))
            t_feat = self.linear3(t_feat)
            t_feat = t_feat.unsqueeze(0).repeat(batch_size,1,1)
            input = input.unsqueeze(1).repeat(1, self.num_basis,1)
            return self.cos_sim(input, t_feat)

        return input
        
        
class SRNsModel(nn.Module):
    def __init__(self,
                 latent_dim,
                 tracing_steps,
                 mode,
                 ratio,
                 has_params=False,
                 use_unet_renderer=False,
                 freeze_networks=False,
                 ):
        super().__init__()

        self.latent_dim = latent_dim
        self.has_params = has_params

        self.num_hidden_units_phi = 256
        self.phi_layers = 4  # includes the in and out layers
        self.rendering_layers = 5  # includes the in and out layers
        self.sphere_trace_steps = tracing_steps
        self.freeze_networks = freeze_networks
        self.mode = mode
        self.ratio = ratio
        
        resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.pre_phi = nn.Linear(in_features=512, out_features = self.latent_dim)

        # Auto-decoder: each scene instance gets its own code vector z
        # self.latent_codes = nn.Embedding(num_instances, latent_dim).cuda()
        # nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=self.latent_dim,
                                                hyper_num_hidden_layers=1,
                                                hyper_hidden_ch=self.latent_dim,
                                                hidden_ch=self.num_hidden_units_phi,
                                                num_hidden_layers=self.phi_layers - 2,
                                                in_ch=3,
                                                out_ch=self.num_hidden_units_phi)

        self.ray_marcher = custom_layers.Raymarcher(num_feature_channels=self.num_hidden_units_phi,
                                                    raymarch_steps=self.sphere_trace_steps)

        if use_unet_renderer:
            self.pixel_generator = custom_layers.DeepvoxelsRenderer(nf0=32, in_channels=self.num_hidden_units_phi,
                                                                    input_resolution=128, img_sidelength=128)
        else:
            self.pixel_generator = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi,
                                                               num_hidden_layers=self.rendering_layers - 1,
                                                               in_features=self.num_hidden_units_phi,
                                                               out_features=3,
                                                               outermost_linear=True)

        if self.freeze_networks:
            all_network_params = (list(self.pixel_generator.parameters())
                                  + list(self.ray_marcher.parameters())
                                  + list(self.hyper_phi.parameters()))
            for param in all_network_params:
                param.requires_grad = False

        # Losses
        self.l2_loss = nn.MSELoss(reduction="mean")

        # List of logs
        self.logs = list()

        print(self)
        print("Number of parameters:")
        util.print_network(self)

    def get_regularization_loss(self, prediction, ground_truth):
        """Computes regularization loss on final depth map (L_{depth} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: Regularization loss on final depth map.
        """
        _, depth = prediction

        neg_penalty = (torch.min(depth, torch.zeros_like(depth)) ** 2)
        return torch.mean(neg_penalty) * 10000

    def get_image_loss(self, prediction, ground_truth):
        """Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()

        loss = self.l2_loss(pred_imgs, trgt_imgs)
        return loss

    def get_latent_loss(self):
        """Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        """
        self.latent_reg_loss = torch.mean(self.embedding ** 2)

        return self.latent_reg_loss

    def get_psnr(self, prediction, ground_truth):
        """Compute PSNR of model image predictions.

        :param prediction: Return value of forward pass.
        :param ground_truth: Ground truth.
        :return: (psnr, ssim): tuple of floats
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()
        batch_size = pred_imgs.shape[0]

        if not isinstance(pred_imgs, np.ndarray):
            pred_imgs = util.lin2img(pred_imgs).detach().cpu().numpy()

        if not isinstance(trgt_imgs, np.ndarray):
            trgt_imgs = util.lin2img(trgt_imgs).detach().cpu().numpy()

        psnrs, ssims = list(), list()
        for i in range(batch_size):
            p = pred_imgs[i].squeeze().transpose(1, 2, 0)
            trgt = trgt_imgs[i].squeeze().transpose(1, 2, 0)

            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5

            ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

            psnrs.append(psnr)
            ssims.append(ssim)

        return psnrs, ssims

    def get_comparisons(self, model_input, prediction, ground_truth=None):
        predictions, depth_maps = prediction

        batch_size = predictions.shape[0]

        # Parse model input.
        intrinsics = model_input["intrinsics"].cuda()
        uv = model_input["uv"].cuda().float()

        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = depth_maps.view(batch_size, -1)

        normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
        normals = F.pad(normals, pad=(1, 1, 1, 1), mode="constant", value=1.)

        predictions = util.lin2img(predictions)

        if ground_truth is not None:
            trgt_imgs = ground_truth["rgb"]
            trgt_imgs = util.lin2img(trgt_imgs)

            return torch.cat((normals.cpu(), predictions.cpu(), trgt_imgs.cpu()), dim=3).detach().numpy()
        else:
            return torch.cat((normals.cpu(), predictions.cpu()), dim=3).detach().numpy()

    def get_output_img(self, prediction):
        pred_imgs, _ = prediction
        return util.lin2img(pred_imgs)

    def write_updates(self, writer, predictions, ground_truth, iter, prefix=""):
        """Writes tensorboard summaries using tensorboardx api.

        :param writer: tensorboardx writer object.
        :param predictions: Output of forward pass.
        :param ground_truth: Ground truth.
        :param iter: Iteration number.
        :param prefix: Every summary will be prefixed with this string.
        """
        predictions, depth_maps = predictions
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()

        batch_size, num_samples, _ = predictions.shape

        # Module"s own log
        for type, name, content, every_n in self.logs:
            name = prefix + name

            if not iter % every_n:
                if type == "image":
                    writer.add_image(name, content.detach().cpu().numpy(), iter)
                    writer.add_scalar(name + "_min", content.min(), iter)
                    writer.add_scalar(name + "_max", content.max(), iter)
                elif type == "figure":
                    writer.add_figure(name, content, iter, close=True)
                elif type == "histogram":
                    writer.add_histogram(name, content.detach().cpu().numpy(), iter)
                elif type == "scalar":
                    writer.add_scalar(name, content.detach().cpu().numpy(), iter)
                elif type == "embedding":
                    writer.add_embedding(mat=content, global_step=iter)

        if not iter % 100:
            output_vs_gt = torch.cat((predictions, trgt_imgs), dim=0)
            output_vs_gt = util.lin2img(output_vs_gt)
            writer.add_image(prefix + "Output_vs_gt",
                             torchvision.utils.make_grid(output_vs_gt,
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

            rgb_loss = ((predictions.float().cuda() - trgt_imgs.float().cuda()) ** 2).mean(dim=2, keepdim=True)
            rgb_loss = util.lin2img(rgb_loss)

            fig = util.show_images([rgb_loss[i].detach().cpu().numpy().squeeze()
                                    for i in range(batch_size)])
            writer.add_figure(prefix + "rgb_error_fig",
                              fig,
                              iter,
                              close=True)

            depth_maps_plot = util.lin2img(depth_maps)
            with torch.no_grad():
                writer.add_image(prefix + "pred_depth",
                                torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                                            scale_each=True,
                                                            normalize=True).cpu().detach().numpy(),
                                iter)

        writer.add_scalar(prefix + "out_min", predictions.min(), iter)
        writer.add_scalar(prefix + "out_max", predictions.max(), iter)

        writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
        writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)

        if iter:
            writer.add_scalar(prefix + "latent_reg_loss", self.latent_reg_loss, iter)

    def forward(self, input, weighted_ebd=None):
        self.logs = list() # log saves tensors that"ll receive summaries when model"s write_updates function is called

        # Parse model input.
        instance_idcs = input["instance_idx"].long().cuda()
        pose = input["pose"].cuda()
        intrinsics = input["intrinsics"].cuda()
        uv = input["uv"].cuda().float()
        input_img = input["input_img"].cuda()
        gt_path = input["gt_path"][0]

        
        if self.has_params: # If each instance has a latent parameter vector, we"ll use that one.
            if weighted_ebd is None:
                self.embedding = input["param"].cuda()
            else:
                self.embedding = weighted_ebd
        else: # Else, we"ll use the embedding.
            if weighted_ebd is not None:
                self.embedding = weighted_ebd
            else:
                self.embedding = self.pre_phi(self.resnet(input_img).squeeze())

        # Interpolation
        '''
        np.random.seed(instance_idcs.item())
        # category = "bed"
        categories = os.listdir("/work/eva0856121/Augmentation/datasets/MN40_VI_imgfilter_10views")
        category_embed2 = np.random.choice(categories, 1)[0]
        
        embedding2_path_list = sorted(list(glob.glob("/work/eva0856121/Augmentation/datasets/MN40_VI_imgfilter_10views/%s/*.pcd" %(category_embed2)))) 
        embedding2_path = np.random.choice(embedding2_path_list, 1)[0]
        embedding2_idcs = embedding2_path.split('/')[-1].split('.')[0]
        embedding2_path = "/work/eva0856121/NVS/code/scene-representation-networks/3D_points/modelnet_all_VI_imgfilter/all_train/" + str(int(embedding2_idcs)) +"_"+ gt_path.split('/')[-1][:-4] + "_embedding.npy"

        embedding2 = torch.from_numpy(np.load(embedding2_path)).cuda().float()
        self.embedding = (7.0 * self.embedding + 3.0 * embedding2) / 10.0
        '''
        
        
        phi = self.hyper_phi(self.embedding) # Forward pass through hypernetwork yields a (callable) SRN.

        # Raymarch SRN phi along rays defined by camera pose, intrinsics and uv coordinates.
        points_xyz, depth_maps, log = self.ray_marcher(cam2world=pose,
                                                       intrinsics=intrinsics,
                                                       uv=uv,
                                                       phi=phi)
        self.logs.extend(log)

        # Sapmle phi a last time at the final ray-marched world coordinates.
        v = phi(points_xyz)


        # Translate features at ray-marched world coordinates to RGB colors.
        novel_views = self.pixel_generator(v)

        # Create image mask (imgfilter)
        '''
        mask = (torch.mean(input["rgb"], 2) != 1)
        points_xyz_save = points_xyz[mask]
        novel_views_save = novel_views[mask]
        '''
        
        # Inference time - Save 3D information (open in testing time)
        # '''
        if self.mode =="test":
            pc_dir_path = os.path.join(f"/eva_data_0/augmentation_output/SRN/test/real_{self.ratio}/3D_info") #, "%s_train" %(category))
            os.makedirs(pc_dir_path, exist_ok=True)
            pc_path = pc_dir_path + '/' + str(instance_idcs.item()) + '_' + gt_path.split('/')[-1][:-4]
            np.save( pc_path + "_embedding.npy", self.embedding.cpu().numpy())
            np.save(pc_path + "_points.npy", points_xyz.cpu().numpy())
            np.save(pc_path + "_rgb.npy", novel_views.cpu().numpy())
            with open(os.path.join(pc_dir_path, "inter_ID.txt"), "a") as fp:
                # print("%06d %06d %s" %(instance_idcs.item(), int(embedding2_idcs), category_embed2), file=fp)
                print("%06d %06d %s" %(instance_idcs.item(), int(instance_idcs), str(instance_idcs)), file=fp)
        # '''

        # Calculate normal map
        with torch.no_grad():
            batch_size = uv.shape[0]
            x_cam = uv[:, :, 0].view(batch_size, -1)
            y_cam = uv[:, :, 1].view(batch_size, -1)
            z_cam = depth_maps.view(batch_size, -1)

            normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            self.logs.append(("image", "normals",
                              torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))
        
        
        return novel_views, depth_maps
