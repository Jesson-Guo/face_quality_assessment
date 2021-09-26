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
            if file_name.endswith('jpg') or file_name.endswith('JPG'):
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
    tensor_shape = tensor_vec['tensorShape']
    out_eul = np.frombuffer(base64.b64decode(data_str), dtype=np.float32) * 90
    out_eul = np.reshape(out_eul, tensor_shape)
    print("---------------------------out_eul---------------------------")
    print(out_eul)
    print()
    print(out_eul.shape)
    print("-----------------------------------------------------------------")

    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][1]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    heatmap = np.frombuffer(base64.b64decode(data_str), dtype=np.float32)
    heatmap = np.reshape(heatmap, tensor_shape)
    print("---------------------------heatmap---------------------------")
    print(heatmap)
    print()
    print(heatmap.shape)
    print("-----------------------------------------------------------------")
    return out_eul, heatmap


def read_ground_truth(img_path):
    txt_path = ""
    if img_path.endswith('jpg'):
        txt_path = img_path.replace('jpg', 'txt')
    elif img_path.endswith('JPG'):
        txt_path = img_path.replace('JPG', 'txt')
    else:
        print("[ERROR], image format is invalid, REQUIRED .jpg")
        exit(0)
    if os.path.exists(txt_path):
        euler_kps_do = True
        img_ori = cv2.imread(img_path)
        x_length = img_ori.shape[1]
        y_length = img_ori.shape[0]
        txt_line = open(txt_path).readline()
        # [YAW] [PITCH] [ROLL]
        eulers_txt = txt_line.strip().split(" ")[:3]
        kp_list = [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
        box_cur = txt_line.strip().split(" ")[3:]
        bndbox = []
        for index in range(len(box_cur) // 2):
            bndbox.append([box_cur[index * 2], box_cur[index * 2 + 1]])
        kp_id = -1
        for box in bndbox:
            kp_id = kp_id + 1
            x_coord = float(box[0])
            y_coord = float(box[1])
            if x_coord < 0 or y_coord < 0:
                continue
            # 修改坐标值，与图片resize后对应
            kp_list[kp_id][0] = int(float(x_coord) / x_length * 96)
            kp_list[kp_id][1] = int(float(y_coord) / y_length * 96)
        return eulers_txt, kp_list, euler_kps_do
    else:
        euler_kps_do = False
        return None, None, euler_kps_do


def get_infer_info(eulers, heatmap):
    kp_coord_ori = list()
    for i, _ in enumerate(heatmap):
        map_1 = heatmap[i].reshape(1, 48 * 48)
        # softmax
        map_1 = np.exp(map_1) / np.sum(np.exp(map_1), axis=1)
        kp_coor = map_1.argmax()
        # 由于最开始将(48,48)转换成了(1, 48*48)，所以这一步是在求原本的位置
        kp_coor = int((kp_coor % 48) * 2.0), int((kp_coor / 48) * 2.0)
        kp_coord_ori.append(kp_coor)
    return kp_coord_ori, eulers


def save_output(out, output_dir, img_path):
    if not output_dir:
        return None
    file_name = img_path.strip().split(os.path.sep)[-1].split('.')[0]
    infer_image_path = os.path.join(output_dir, f"{file_name}_infer.txt")
    output = open(infer_image_path, 'w')
    for item in out:
        output.write(str(item))


def main(args):
    kp_error_all = [[], [], [], [], []]
    eulers_error_all = [[], [], []]
    kp_ipn = []
    path = args.dataset

    stream_manager_api = get_stream_manager(args.pipeline)
    for img_path in get_dataset(path):
        # [out_euler, out_kps] (1,3); (1,5,48,48)
        out_eul, heatmap = do_infer_image(stream_manager_api, img_path)
        save_output([out_eul, heatmap], args.output, img_path)
        # eulers_gt: 角度信息
        # kp_list:   5个关键点的坐标
        eulers_gt, kp_list, euler_kps_do = read_ground_truth(img_path)
        if not euler_kps_do:
            continue

        kp_coord_ori, eulers_ori = get_infer_info(out_eul, heatmap)
        # 计算error
        eulgt = list(eulers_gt)
        for euler_id, _ in enumerate(eulers_ori):
            eulers_error_all[euler_id].append(abs(eulers_ori[euler_id] - float(eulgt[euler_id])))
        eye01 = kp_list[0]
        eye02 = kp_list[1]
        eye_dis = 1
        cur_flag = True
        if eye01[0] < 0 or eye01[1] < 0 or eye02[0] < 0 or eye02[1] < 0:
            cur_flag = False
        else:
            eye_dis = np.sqrt(np.square(abs(eye01[0] - eye02[0])) + np.square(abs(eye01[1] - eye02[1])))
        cur_error_list = []
        for i in range(5):
            if kp_list[i][0] != -1:
                dis = np.sqrt(np.square(kp_list[i][0] - kp_coord_ori[i][0]) + np.square(kp_list[i][1] - kp_coord_ori[i][1]))
                kp_error_all[i].append(dis)
                cur_error_list.append(dis)
        if cur_flag:
            kp_ipn.append(sum(cur_error_list) / len(cur_error_list) / eye_dis)

    kp_ave_error = []
    for kps, _ in enumerate(kp_error_all):
        kp_ave_error.append("%.3f" % (sum(kp_error_all[kps]) / len(kp_error_all[kps])))

    euler_ave_error = []
    elur_mae = []
    for eulers, _ in enumerate(eulers_error_all):
        euler_ave_error.append("%.3f" % (sum(eulers_error_all[eulers]) / len(eulers_error_all[eulers])))
        elur_mae.append((sum(eulers_error_all[eulers]) / len(eulers_error_all[eulers])))

    print('========== 5 keypoints average err:' + str(kp_ave_error))
    print('========== 3 eulers average err:' + str(euler_ave_error))
    print('========== IPN of 5 keypoints:' + str(sum(kp_ipn) / len(kp_ipn) * 100))
    print('========== MAE of elur:' + str(sum(elur_mae) / len(elur_mae)))
    stream_manager_api.DestroyAllStreams()


if __name__ == "__main__":
    args = parse_arg()
    main(args)
