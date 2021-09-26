import argparse
import os

import time
import moxing as mox
import zipfile
import export
import train
from model_utils.config import config


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 模型输出目录
    parser.add_argument('--train_url', type=str, default='', help='the path model saved')

    # 数据集目录
    parser.add_argument('-d', '--data_url', type=str, default='', help='the training data')

    # 抽取出来的超参配置
    parser.add_argument("--device_id", type=str, default="0", help="device id")
    parser.add_argument("--train_label_file", type=str, default="", help="training label file")
    parser.add_argument("--eval_dir", type=str, default="", help="evaluation directory")
    return parser.parse_args()


def main():
    '''start script for model export'''
    args = parse_args()
    print("Training setting:", args)
    os.environ["DEVICE_ID"] = args.device_id
    # 开始训练
    train.run_train()
    # 拷贝数据集到cache目录
    os.makedirs("/cache/checkpoint_path", exist_ok=True)
    print(args.train_url + "output")
    mox.file.copy_parallel("/cache/train/output", "/cache/checkpoint_path")
    # config.checkpoint_url = os.path.join(args.train_url + "output")
    print("模型转换")
    print(config.train_url)
    t = config.train_url.split("/")
    t[-1] = "output"
    config.train_url = ""
    for s in t:
        config.train_url += s + "/"

    print(config.train_url)
    # config.train_url = config.train_url[:-6]+"output/"
    export.run_export()


if __name__ == '__main__':
    main()
