import argparse
import base64
import json
import os

import cv2
import numpy as np
from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi


def parse_arg():
    parser = argparse.ArgumentParser(description="FQA infer")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="the directory of dataset")
    parser.add_argument("-p", "--pipeline", type=str, required=True, help="the path of .pipeline file")
    parser.add_argument("-o", "--output", type=str, default="", help="the path of pipeline file")
    return parser.parse_args()


def get_dataset(path):
    for root, dirs, files in os.walk(path):
        for file_name in files:  # 遍历文件夹
            if file_name.endswith('jpg'):
                yield os.path.join(path, file_name)
        break


def get_stream_manager(pipeline_path):
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with open(pipeline_path, 'rb') as f:
        pipeline_content = f.read()

    ret = stream_manager_api.CreateMultipleStreams(pipeline_content)
    if ret != 0:
        print("Failed to create stream, ret=%s" % str(ret))
        exit()
    return stream_manager_api


def do_infer_image(stream_manager_api, image_path):
    stream_name = b'face_quality_assessment'
    data_input = MxDataInput()
    with open(image_path, 'rb') as f:
        data_input.data = f.read()

    unique_id = stream_manager_api.SendData(stream_name, 0, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    infer_result = stream_manager_api.GetResult(stream_name, unique_id)
    if infer_result.errorCode != 0:
        print(f"GetResult error. errorCode={infer_result.errorCode},"
              f"errorMsg={infer_result.data.decode()}")
        exit()

    infer_result_json = json.loads(infer_result.data.decode())
    content = json.loads(infer_result_json['metaData'][0]['content'])
    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][0]
    data_str = tensor_vec['dataStr']
    out_eul = np.reshape(np.frombuffer(base64.b64decode(data_str), dtype=np.float32), tensor_vec['tensorShape'])
    print("---------------------------data---------------------------")
    print(out_eul)
    print("-----------------------------------------------------------------")


def main(args):
    path = args.dataset
    stream_manager_api = get_stream_manager(args.pipeline)
    for img_path in get_dataset(path):
        do_infer_image(stream_manager_api, img_path)
    stream_manager_api.DestroyAllStreams()


if __name__ == "__main__":
    args = parse_arg()
    main(args)
