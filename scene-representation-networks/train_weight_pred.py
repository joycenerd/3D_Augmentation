import configargparse
import os, time, datetime

import torch
import numpy as np

import dataio
from torch.utils.data import DataLoader
from srns import *
import util
from tqdm import tqdm

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Note: in contrast to training, no multi-resolution!
p.add_argument('--img_sidelength', type=int, default=128, required=False,
               help='Sidelength of test images.')

p.add_argument('--data_root', required=True, help='Path to directory with training data.')
p.add_argument('--logging_root', type=str, default='./logs',
               required=False, help='Path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--batch_size', type=int, default=32, help='Batch size.')
p.add_argument('--preload', action='store_true', default=False, help='Whether to preload data to RAM.')

p.add_argument('--max_num_instances', type=int, default=-1,
               help='If \'data_root\' has more instances, only the first max_num_instances are used')
p.add_argument('--specific_observation_idcs', type=str, default=None,
               help='Only pick a subset of specific observations for each instance.')
p.add_argument('--has_params', action='store_true', default=False,
               help='Whether each object instance already comes with its own parameter vector.')

p.add_argument('--save_out_first_n',type=int, default=250, help='Only saves images of first n object instances.')
p.add_argument('--checkpoint_path', default=None, help='Path to trained model.')

# Model options
p.add_argument('--tracing_steps', type=int, default=10, help='Number of steps of intersection tester.')
p.add_argument('--fit_single_srn', action='store_true', required=False,
               help='Only fit a single SRN for a single scene (not a class of SRNs) --> no hypernetwork')
p.add_argument('--use_unet_renderer', action='store_true',
               help='Whether to use a DeepVoxels-style unet as rendering network or a per-pixel 1x1 convnet')
p.add_argument('--embedding_size', type=int, default=256,
               help='Dimensionality of latent embedding.')
p.add_argument("--num_basis", type=int, required=True, help="Number of basis objs are used")
p.add_argument("--gpu", type=str, required=True, help="Choose GPU number")

opt = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device('cuda')

util.cond_mkdir(opt.logging_root)
# Save command-line parameters to log directory.
with open(os.path.join(opt.logging_root, "params.txt"), "w") as out_file:
    out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

def back_up():
    backup_dir = os.path.join(opt.logging_root, 'backup')
    util.cond_mkdir(backup_dir)
    os.system("cp train_weight_pred.py %s/" %(backup_dir))
    os.system("cp srns.py %s/" %(backup_dir))
    os.system("cp dataio.py %s/" %(backup_dir))

def create_dataloader():
    if opt.specific_observation_idcs is not None:
        specific_observation_idcs = list(map(int, opt.specific_observation_idcs.split(',')))
    else:
        specific_observation_idcs = None

    dataset = dataio.SceneClassDataset(root_dir=opt.data_root,
                                       max_num_instances=opt.max_num_instances,
                                       specific_observation_idcs=None,
                                       max_observations_per_instance=10,
                                       samples_per_instance=1,
                                       img_sidelength=opt.img_sidelength,)
    dataloader = DataLoader(dataset,
                         batch_size=16,
                         shuffle=True,
                         drop_last=True,
                         num_workers=6)

    test_dataset = dataio.SceneClassDataset(root_dir="/work/eva0856121/datasets/srn_data/cars_test/",
                                       max_num_instances=-1,
                                       specific_observation_idcs=specific_observation_idcs,
                                       max_observations_per_instance=-1,
                                       samples_per_instance=1,
                                       img_sidelength=opt.img_sidelength,)
    test_dataloader = DataLoader(test_dataset,
                         batch_size=16,
                         shuffle=False,
                         drop_last=False,
                         num_workers=6)

    return dataset, dataloader, test_dataset, test_dataloader

def create_network(USE_EMBED):
    model = SRNsModel(latent_dim=opt.embedding_size,
                      has_params=opt.has_params,
                      use_unet_renderer=opt.use_unet_renderer,
                      tracing_steps=opt.tracing_steps).cuda()

    weightPred = AnalogyPred(num_basis= opt.num_basis, USE_EMBED=USE_EMBED, mode="regression").cuda()

    optimizer = torch.optim.Adam(list(weightPred.parameters()) , lr=1e-3)
        
    assert (opt.checkpoint_path is not None), "Have to pass checkpoint!"
    print("Loading SRN model from %s" % opt.checkpoint_path)
    util.custom_load(model, path=opt.checkpoint_path, discriminator=None, overwrite_embeddings=False)
        
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

    return model, weightPred, optimizer 

def train_weightPred(model, weightPred, dataset, dataloader, test_dataset, test_dataloader, optimizer,  USE_EMBED):
    
    fixed_data = dataloader.collate_fn([dataset[0]])
    fixed_input = fixed_data[0][0]
    if USE_EMBED:
        basis_dict = util.read_pickle("./pretrained_model/srn_new/features/cars_train_embedding_all.pkl")
    else:
        basis_dict = util.read_pickle("./pretrained_model/srn_new/features/cars_train_feature_all.pkl")
    
    selected_id = [i for i in range(200)]
    basis_feat = [basis_dict[k].squeeze() for k in selected_id]
    basis_feat = torch.stack(basis_feat).cuda() # 10 x 512 training features
    print("The shape of basis: ", basis_feat.shape)

    print('Begin to train weight prediction branch...')
    for epoch in range(50):
        weightPred.train()
        model.eval()
        psnrs, ssims = list(), list()
        for batch_id, data in enumerate(dataloader):
            model_input, ground_truth = data
            model_input = model_input[0]
            ground_truth = ground_truth[0]
            if USE_EMBED:
                feat_sim = weightPred(model.pre_phi(model.resnet(model_input["input_img"].cuda()).squeeze()), basis_feat)
                pred_output = model(input = model_input, weighted_ebd = torch.matmul(feat_sim, basis_feat.squeeze()))
            else:
                feat_sim = weightPred(model.resnet(model_input["input_img"].cuda()).squeeze(), basis_feat)
                pred_output = model(input = model_input, weighted_ebd = model.pre_phi(torch.matmul(feat_sim, basis_feat.squeeze())))

            optimizer.zero_grad()
            dist_loss = model.get_image_loss(pred_output, ground_truth)
            reg_loss = model.get_regularization_loss(pred_output, ground_truth)
            latent_loss = model.get_latent_loss()

            weighted_dist_loss = 200 * dist_loss
            weighted_reg_loss = 1 * reg_loss
            weighted_latent_loss = 0.001 * latent_loss

            total_loss = weighted_dist_loss + weighted_reg_loss + weighted_latent_loss
            total_loss.backward()
            optimizer.step()

            psnr, ssim = model.get_psnr(pred_output, ground_truth)
            psnrs.extend(psnr)
            ssims.extend(ssim)

            print("Training epoch %d. Running mean PSNR %0.6f SSIM %0.6f" % (epoch, np.mean(psnrs), np.mean(ssims)))
            with open(os.path.join(opt.logging_root, "results.txt"), "a") as out_file:
                out_file.write("Epoch %d. Training Running mean PSNR %0.6f SSIM %0.6f\n" % (epoch, np.mean(psnrs), np.mean(ssims)))
            
            if epoch % 5 == 0:
                util.save_model(model_dir, weightPred, 'analogy', epoch)
                util.save_opt(model_dir, optimizer, 'opt', epoch)

                output_imgs = model.get_output_img(pred_output).cpu().detach().numpy()
                comparisons = model.get_comparisons(model_input,
                                                    pred_output,
                                                    ground_truth)
                for idx in range(1):
                    img_only_path = os.path.join(train_img_dir, "output", "%03d" % epoch)
                    comp_path = os.path.join(train_img_dir, "compare", "%03d" % epoch)

                    util.cond_mkdir(img_only_path)
                    util.cond_mkdir(comp_path)

                    pred = util.convert_image(output_imgs[idx].squeeze())
                    comp = util.convert_image(comparisons[idx].squeeze())

                    util.write_img(pred, os.path.join(img_only_path, "%03d_%06d.png" % (batch_id, idx)))
                    util.write_img(comp, os.path.join(comp_path, "%03d_%06d.png" % (batch_id, idx)))
        
        if epoch % 5 == 0:
            weightPred.eval()
            model.eval()
            with torch.no_grad():
                print('Testing')
                psnrs, ssims = list(), list()
                for batch_id, data in enumerate(test_dataloader):
                    model_input, ground_truth = data
                    model_input = model_input[0]
                    ground_truth = ground_truth[0]
                    if USE_EMBED:
                        feat_sim = weightPred(model.pre_phi(model.resnet(model_input["input_img"].cuda()).squeeze()), basis_feat)
                        pred_output = model(input = model_input, weighted_ebd = torch.matmul(feat_sim, basis_feat.squeeze()))
                    else:
                        feat_sim = weightPred(model.resnet(model_input["input_img"].cuda()).squeeze(), basis_feat)
                        pred_output = model(input = model_input, weighted_ebd = model.pre_phi(torch.matmul(feat_sim, basis_feat.squeeze())))

                    psnr, ssim = model.get_psnr(pred_output, ground_truth)
                    psnrs.extend(psnr)
                    ssims.extend(ssim)

                    print("Testing epoch %d. Running mean PSNR %0.6f SSIM %0.6f" % (epoch, np.mean(psnrs), np.mean(ssims)))
                    with open(os.path.join(opt.logging_root, "results.txt"), "a") as out_file:
                        out_file.write("Epoch %d. Testing Running mean PSNR %0.6f SSIM %0.6f\n" % (epoch, np.mean(psnrs), np.mean(ssims)))
                    
                    output_imgs = model.get_output_img(pred_output).cpu().detach().numpy()
                    comparisons = model.get_comparisons(model_input,
                                                        pred_output,
                                                        ground_truth)
                    for idx in range(len(output_imgs)):
                        img_only_path = os.path.join(test_img_dir, "output", "%03d" % epoch)
                        comp_path = os.path.join(test_img_dir, "compare", "%03d" % epoch)

                        util.cond_mkdir(img_only_path)
                        util.cond_mkdir(comp_path)

                        pred = util.convert_image(output_imgs[idx].squeeze())
                        comp = util.convert_image(comparisons[idx].squeeze())

                        util.write_img(pred, os.path.join(img_only_path, "%03d_%06d.png" % (batch_id, idx)))
                        util.write_img(comp, os.path.join(comp_path, "%03d_%06d.png" % (batch_id, idx)))

    
if __name__ == '__main__':
    back_up()
    USE_EMBED = True

    # Create recording place
    model_dir = os.path.join(opt.logging_root, 'model')
    util.cond_mkdir(model_dir)
    train_img_dir = os.path.join(opt.logging_root, 'train_image')
    util.cond_mkdir(train_img_dir)
    test_img_dir = os.path.join(opt.logging_root, 'test_image')
    util.cond_mkdir(test_img_dir)
    # Get model, dataset ready 
    dataset, dataloader, test_dataset, test_dataloader= create_dataloader()
    print(len(dataset))
    print(len(test_dataset))
    model, weightPred, optimizer = create_network(USE_EMBED)

    train_weightPred(model, weightPred, dataset, dataloader, test_dataset, test_dataloader, optimizer, USE_EMBED)
