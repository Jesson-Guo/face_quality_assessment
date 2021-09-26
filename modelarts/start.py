"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
"""
import argparse
import os

import moxing as mox
import export
import train
from model_utils.config import config


def parse_args():
    """get the cmd input args"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_url', type=str, default='', help='the path model saved')
    parser.add_argument('-d', '--data_url', type=str, default='', help='the training data')

    parser.add_argument("--device_id", type=str, default="0", help="device id")
    parser.add_argument("--train_label_file", type=str, default="", help="training label file")
    parser.add_argument("--eval_dir", type=str, default="", help="evaluation directory")
    return parser.parse_args()


def main():
    """start script for model training and exporting"""
    args = parse_args()
    print("Training setting:", args)
    os.environ["DEVICE_ID"] = args.device_id
    train.run_train()
    os.makedirs("/cache/checkpoint_path", exist_ok=True)
    mox.file.copy_parallel("/cache/train/output", "/cache/checkpoint_path")
    t = config.train_url.split("/")
    t[-1] = "output"
    config.train_url = ""
    for s in t:
        config.train_url += s + "/"
    export.run_export()


if __name__ == '__main__':
    main()
