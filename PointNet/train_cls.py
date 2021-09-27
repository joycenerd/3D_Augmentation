"""
Author: Benny
Date: Nov 2019
Instance accuracy ... classification accuracy
Class accuracy ... consider the number of each class
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import os
import argparse
import logging
import importlib
import sys
import torch
import numpy as np
import provider
from tqdm import tqdm

# Add ./models into system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models"))

# Official
parser = argparse.ArgumentParser("PointNet")
parser.add_argument("--batch_size", type=int, default=24, help="Batch size in training [default: 24]")
parser.add_argument("--model", default="pointnet_cls", help="Model name [default: pointnet_cls]")
parser.add_argument("--epoch",  default=200, type=int, help="Number of epoch in training [default: 200]")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate in training [default: 0.001]")
parser.add_argument("--gpu", type=str, default="0", help="Specify gpu device [default: 0]")
parser.add_argument("--num_point", type=int, default=1024, help="Point Number [default: 1024]")
parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer for training [default: Adam]")
parser.add_argument("--output_dir", type=str, default="/work/eva0856121/Augmentation/code/outputs/PointNet", help="Experiment root")
parser.add_argument("--decay_rate", type=float, default=1e-4, help="Decay rate [default: 1e-4]")
# MSN_PointNet
parser.add_argument("--sparsify_mode", type=str, default="random", choices=["PN", "random", "fps", "zorder", "multizorder"], help="Sparsify mode")
parser.add_argument("--dataset_mode", type=str, default="ModelNet40", choices=["ModelNet40", "ModelNet10", "ShapeNet", "ShapeNet_all"], help="Dataset mode. PLZ choose in [ModelNet40, ModelNet10, ShapeNet, ShapeNet_all]")
parser.add_argument("--zorder_mode", type=str, default="keep", choices=["keep", "discard"], help="Zorder sampled mode. PLZ choose in [keep, discard]")
parser.add_argument("--trans_feat", action="store_true", default=False, help='Whether to use transform feature')
args = parser.parse_args()

# HYPER PARAMETER
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # GPU devices
num_classes = {"ModelNet40": 40, "ModelNet10": 10, "ShapeNet": 8, "ShapeNet_all": 57}
num_class = num_classes[args.dataset_mode] # Number of class (default for ModelNet40)

train_save = {
    "instance_acc": [],
    "loss": []
} # Save training accuracy and loss
test_save = {
    "instance_acc": [],
    "class_acc": []
} # Save testing instance accuracy and class accuracy

def create_output_dir():
    # Create output directry according to sparsify mode, normalize, trans_feat
    trans_feat_dir = "_transfeat" if args.trans_feat else ""
    mode_dir = args.sparsify_mode + trans_feat_dir
    if args.sparsify_mode == "zorder" or args.sparsify_mode == "multizorder":
        mode_dir = args.zorder_mode + "_" + mode_dir
    if args.output_dir == "/work/eva0856121/Augmentation/code/outputs/PointNet":
        # output_dir = os.path.join(args.output_dir, args.dataset_mode + "_cls", mode_dir)
        output_dir = os.path.join(args.output_dir, "aug_" + args.dataset_mode + "_cls", mode_dir + "_MN40_VI_inter_rotate90AugOurs")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def set_logger(log_dir):
    # Setup LOG file format
    global logger
    logger = logging.getLogger(args.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, args.model + ".txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_string(message):
    # Write message into log.txt
    logger.info(message)
    print(message)

def backup_python_file(backup_dir):
    os.system("cp ./train_cls.py {}".format(backup_dir))
    os.system("cp ./models/{}.py {}".format(args.model, backup_dir))
    os.system("cp ./models/pointnet_util.py {}".format(backup_dir))

def create_dataloader():
    print("Load " + args.dataset_mode + " as dataset ...")

    # Create training dataloader
    TRAIN_DATASET = ModelNetDataLoader(npoint=args.num_point, split="train", sparsify_mode=args.sparsify_mode, dataset_mode=args.dataset_mode, zorder_mode=args.zorder_mode)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # Create testing dataloader
    TEST_DATASET = ModelNetDataLoader(npoint=args.num_point, split="test", sparsify_mode=args.sparsify_mode, dataset_mode=args.dataset_mode, zorder_mode=args.zorder_mode)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainDataLoader, testDataLoader

def create_network():
    # Create network (classifier) and criterion
    MODEL = importlib.import_module(args.model) # Load model from args.model (e.g. pointnet_cls.py)
    classifier = MODEL.get_model(num_class).cuda()
    criterion = MODEL.get_loss(trans_feat_switch=args.trans_feat).cuda()

    # Try load pretrained weights
    try:
        checkpoint = torch.load(os.path.join(checkpoints_dir, "best_model.pth"))
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])

        log_string("Use pretrain model")
    except:
        log_string("No existing model, starting training from scratch...")
        start_epoch = 0


    # Fix encoder initial weight
    # for child in classifier.feat.children():
    #     for param in child.parameters():
    #         param.requires_grad = False

    # Setup optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # Setup scheduler for optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    return classifier, criterion, optimizer, scheduler, start_epoch

def train(classifier, trainDataLoader, optimizer, scheduler, criterion):
    # TRAIN MODE
    mean_correct = []
    scheduler.step()
    for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        # Get points and target from trainDataLoader
        points, target = data
        points = points.data.numpy()

        # Do something like augmentation
        points = provider.random_point_dropout(points)
        points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
        points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
        points = torch.Tensor(points)
        target = target[:, 0]

        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()

        # Start training
        classifier = classifier.train()
        pred, trans_feat = classifier(points)
        loss = criterion(pred, target.long(), trans_feat)
        train_save["loss"].append(loss.item()) # Save training loss
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

        loss.backward()
        optimizer.step()

    train_instance_acc = np.mean(mean_correct)
    train_save["instance_acc"].append(train_instance_acc) # Save training instance accuracy

    return train_instance_acc

def test(model, testDataLoader):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    # print(model.feat.conv1.weight[0][0])
    for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
        # Get points and target from testDataLoader
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        # Evaluate by PointNet model
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1] # prediction results

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0]) # Compute accuracy of certain class
            class_acc[cat, 1] += 1 # Compute number of certain class
        correct = pred_choice.eq(target.long().data).cpu().sum() # Total number of correct results
        mean_correct.append(correct.item() / float(points.size()[0])) # Mean instance accuracy within one batch size
    
    class_acc[:, 2] =  class_acc[:, 0] / class_acc[:, 1] # The class accuracy of each class
    class_acc = np.mean(class_acc[:, 2]) # Mean class accuracy (all objects)
    instance_acc = np.mean(mean_correct) # Mean instance accuracy (all objects)

    # Save testing accuracy (instance and class)
    test_save["instance_acc"].append(instance_acc)
    test_save["class_acc"].append(class_acc)

    return instance_acc, class_acc

if __name__ == "__main__":
    # Create output direcotry
    output_dir = create_output_dir()
    backup_dir = os.path.join(output_dir, "backup")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Backup important .py file
    backup_python_file(backup_dir)

    # Setup LOG file format
    set_logger(log_dir)
    log_string("Argument parameter: {}".format(args))

    # Create training and testing dataloader
    trainDataLoader, testDataLoader = create_dataloader()

    # Create network (classifier), optimizer, scheduler
    classifier, criterion, optimizer, scheduler, start_epoch = create_network()

    # Setup parameters for training and testing
    global_epoch = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    # Start training
    logger.info("Start training...")
    for epoch in range(start_epoch, args.epoch):
        log_string("Epoch %d (%d/%s):" % (global_epoch+1, epoch+1, args.epoch))

        # TRAIN MODE
        train_instance_acc = train(classifier, trainDataLoader, optimizer, scheduler, criterion)
        log_string("Train Instance Accuracy: %f" % (train_instance_acc))

        # TEST MODE
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if class_acc >= best_class_acc:
                best_class_acc = class_acc

            log_string("Test Instance Accuracy: %f, Class Accuracy: %f" % (instance_acc, class_acc))
            log_string("Best Instance Accuracy: %f, Class Accuracy: %f" % (best_instance_acc, best_class_acc))

        # Save best training details
        if instance_acc >= best_instance_acc:
            logger.info("Save model...")
            save_path = os.path.join(checkpoints_dir, "best_model.pth")
            log_string("Saving at %s" % (save_path))
            state = {
                "epoch": best_epoch,
                "instance_acc": instance_acc,
                "class_acc": class_acc,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, save_path)
        global_epoch += 1

        # Save weights and [training, testing] results
        # if epoch % 5 == 0:
        #     torch.save(state, os.path.join(checkpoints_dir, "model_%d.pth" %(epoch)))
        np.save(os.path.join(output_dir, "train_save.npy"), train_save)
        np.save(os.path.join(output_dir, "test_save.npy"), test_save)

    logger.info("End of training...")
