import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--sparsify_mode", type=str, default="PN", help="Sparsify mode")
parser.add_argument("--trans_feat", action="store_true", default=False, help='Whether to use transform feature')
parser.add_argument("--root_dir", type=str, default="/eva_data/psa/code/outputs/PointNet/classification", help="The result directory of PointNet.")
parser.add_argument("--output_dir", type=str, default="/eva_data/psa/code/outputs/PointNet/classification/chart", help="The output directory of PointNet loss / accuracy charts.")
args = parser.parse_args()

def show_results(result, category):
    plt.figure(figsize=(10, 6))
    plt.title("PointNet: " + args.sparsify_mode.replace(",", ", ") + "\n" + category)
    for label, data in result.items():
        if category == "loss":
            plt.plot(np.arange(0, len(data)/100), data[::100], label=label)
        else:
            plt.plot(np.arange(0, len(data)), data, label=label + ": max_acc=%.2f" % (np.max(data) * 100))
    if category == "loss":
        plt.xlabel("per 100 iterations")
    else:
        plt.xlabel("epoch")
    plt.ylabel(category)
    plt.legend()
    os.makedirs(os.path.join(args.output_dir, args.sparsify_mode.replace(",", "_")), exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, args.sparsify_mode.replace(",", "_"), category.replace(" ", "_") + ".png"))
    plt.close()

if __name__ == "__main__":
    train_save, test_save = {}, {}
    mode_dirs = os.listdir(args.root_dir)
    sparsify_modes = args.sparsify_mode.split(",")
    for sparsify_mode in sparsify_modes:
        for mode_dir in mode_dirs:
            if sparsify_mode in mode_dir.split("_")[0]:
                if (not args.trans_feat) or ((args.trans_feat) and ("transfeat" in mode_dir)):
                    train_save[mode_dir] = np.load(os.path.join(args.root_dir, mode_dir, "train_save.npy"), allow_pickle=True)
                    test_save[mode_dir] = np.load(os.path.join(args.root_dir, mode_dir, "test_save.npy"), allow_pickle=True)
    
    for key_train, key_test in zip(train_save.keys(), test_save.keys()):
        train_save[key_train] = train_save[key_train].item()
        test_save[key_test] = test_save[key_test].item()

    train_instance_acc = {}
    train_loss = {}
    test_instance_acc = {}
    test_class_acc = {}
    for key_train, key_test in zip(train_save.keys(), test_save.keys()):
        train_instance_acc[key_train] = train_save[key_train]["instance_acc"]
        train_loss[key_train] = train_save[key_train]["loss"]
        test_instance_acc[key_test] = test_save[key_test]["instance_acc"]
        test_class_acc[key_test] = test_save[key_test]["class_acc"]

    show_results(train_instance_acc, "training instance accuracy")
    show_results(train_loss, "loss")
    show_results(test_instance_acc, "testing instance accuracy")
    show_results(test_class_acc, "testing class accuracy")
    # category = "training instance accuracy"
    # plt.figure(figsize=(10, 6))
    # plt.title("PointNet: " + args.sparsify_mode.replace(",", "_") + "\n" + category)
    # for label, data in train_instance_acc.items():
    #     plt.plot(np.arange(0, len(data)), data, label=label)
    # plt.xlabel("epoch")
    # plt.ylabel(category)
    # plt.legend(loc=4)
    # os.makedirs(os.path.join(args.output_dir, args.sparsify_mode.replace(",", "_")))
    # plt.savefig(os.path.join(args.output_dir, args.sparsify_mode.replace(",", "_"), category.replace(" ", "_") + ".png"))
    # plt.close()
        
    # plt.figure()
    # plt.title("PointNet " + mode + " - " + category)
    # for result in [train_save, test_save]:
    #     for transfeat_key in result.keys():
    #         content = result[transfeat_key]
    #         for content_key in content.keys():
    #             data = content[content_key]
                

    # train_instance_acc = np.load(os.path.join(args.result_dir, "train_instance_accuracy.npy"))
    # train_loss = np.load(os.path.join(args.result_dir, "train_loss.npy"))
    # test_instance_acc = np.load(os.path.join(args.result_dir, "test_instance_accuracy.npy"))
    # test_class_acc = np.load(os.path.join(args.result_dir, "test_class_accuracy.npy"))

    # show_results("loss", train_loss, "pretrained_fixed", "training loss", "train_loss.png")
    # show_results("accuracy", train_instance_acc, "pretrained_fixed", "training instance accuracy", "train_instance_accuracy.png")
    # show_results("accuracy", test_instance_acc, "pretrained_fixed", "testing instance accuracy", "test_instance_accuracy.png")
    # show_results("accuracy", test_class_acc, "pretrained_fixed", "testing class accuracy", "test_class_accuracy.png")
