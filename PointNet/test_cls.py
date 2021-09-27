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
from tqdm import tqdm

# Add ./models into system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models"))

# Official
parser = argparse.ArgumentParser("PointNet")
parser.add_argument("--batch_size", type=int, default=24, help="Batch size in training")
parser.add_argument("--gpu", type=str, default='1', help="specify gpu device")
parser.add_argument("--num_point", type=int, default=1024, help="Point Number [default: 1024]")
parser.add_argument("--output_dir", type=str, required=True, help="PLZ input [/eva_data/psa/code/outputs/PointNet/classification/???")
parser.add_argument("--num_votes", type=int, default=3, help="Aggregate classification scores with voting [default: 3]")
args = parser.parse_args()

# HYPER PARAMETER
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # GPU devices
num_class = 40 #10 #40 # Number of class (default for ModelNet40)
np.random.seed(1234)
torch.manual_seed(1234)

def set_logger(model_name, log_dir):
    # Setup LOG file format
    global logger
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, "eval_ours.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_string(message):
    # Write message into log.txt
    logger.info(message)
    print(message)

def create_dataloader(sparsify_mode):
    log_string("Load dataset ...")

    # Create testing dataloader
    TEST_DATASET = ModelNetDataLoader(npoint=args.num_point, split="test", sparsify_mode=sparsify_mode)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return testDataLoader

def create_network(model_name, checkpoints_dir):
    # Create network (classifier)
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_class).cuda()
    # print(classifier.state_dict().keys())

    # load pretrained weights
    checkpoint = torch.load(os.path.join(checkpoints_dir, "best_model.pth"))
    classifier.load_state_dict(checkpoint["model_state_dict"])

    return classifier

def test(model, testDataLoader, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
        # Get points and target from testDataLoader
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        
        # Evaluate by PointNet model
        classifier = model.eval()
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()
        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1] # prediction results
        log_string("pred: {}".format(pred_choice))
        log_string("target: {}".format(target))

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0]) # Compute accuracy of certain class
            class_acc[cat, 1] +=1 # Compute number of certain class
        correct = pred_choice.eq(target.long().data).cpu().sum() # Total number of correct results
        mean_correct.append(correct.item() / float(points.size()[0])) # Mean instance accuracy within one batch size

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1] # The class accuracy of each class
    log_string("All class accuracy:")
    for idx, acc in enumerate(class_acc[:, 2]):
        log_string("{:02d}: {:.2f}".format(idx, acc*100))
    class_acc = np.mean(class_acc[:, 2]) # Mean class accuracy (all objects)
    instance_acc = np.mean(mean_correct) # Mean instance accuracy (all objects)
    
    return instance_acc, class_acc

if __name__ == "__main__":
    # Create output direcotry
    mode = args.output_dir.split("/")[-1]
    if len(mode) == 0:
        mode = args.output_dir.split("/")[-2]
    sparsify_mode = mode.split("_")[0]
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")
    log_files = os.listdir(log_dir)
    for log_file in log_files:
        if "pointnet" in log_file:
            model_name = log_file.split(".")[0]

    # Setup LOG file format
    set_logger(model_name, log_dir)
    log_string("Argument parameter: {}".format(args))

    # Create testing dataloader
    # testDataLoader = create_dataloader(sparsify_mode)
    testDataLoader = create_dataloader(sparsify_mode)

    # Create network (classifier)
    classifier = create_network(model_name, checkpoints_dir)
    
    # Start testing
    logger.info("Start testing...")
    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes)
        log_string("Test Instance Accuracy: %f, Class Accuracy: %f" % (instance_acc, class_acc))
