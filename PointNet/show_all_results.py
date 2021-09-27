import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, default="/eva_data/Allen/PointNet_MSN_pretrained/log/classification/pretrained_fixed/results", help="The result directory of PointNet.")
parser.add_argument("--output_dir", type=str, default="./", help="The output directory of PointNet loss / accuracy charts.")
args = parser.parse_args()

if __name__ == "__main__":
    root_dirs = ["PointNet_baseline", "PointNet_baseline", "PointNet_MSN_concat", "PointNet_MSN_concat", "PointNet_MSN_pretrained"]
    result_dirs = ["pointnet_baseline", "no_trans_feat_baseline", "concat", "concat_add_fc", "pretrained_fixed"]

    plt.figure(figsize=(16, 9))
    plt.title("PointNet - testing class accuracy", fontsize=15)

    for root_dir, result_dir in zip(root_dirs, result_dirs):
        # result = np.load(os.path.join("/eva_data/Allen", root_dir, "log/classification", result_dir, "results/train_loss.npy"))
        # plt.plot(np.arange(0, len(result)/100), result[::100], label=result_dir)

        # result = np.load(os.path.join("/eva_data/Allen", root_dir, "log/classification", result_dir, "results/train_instance_accuracy.npy"))
        # result = np.load(os.path.join("/eva_data/Allen", root_dir, "log/classification", result_dir, "results/test_instance_accuracy.npy"))
        result = np.load(os.path.join("/eva_data/Allen", root_dir, "log/classification", result_dir, "results/test_class_accuracy.npy"))
        plt.plot(np.arange(0, len(result)), result, label=result_dir)
        
    plt.xlabel("epoch")
    plt.ylabel("testing class accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "all_test_class_accuracy.png"))
    plt.close()