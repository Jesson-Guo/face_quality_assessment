import json
import os
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # 新建一个流管理StreamManager对象并初始化
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 构建pipeline
    pipeline = {
        "face_quality_assessment": {
            "stream_config": {
                "deviceId": "0"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "4096000"
                },
                "factory": "appsrc",
                "next": "mxpi_imagedecoder0"
            },
            "mxpi_imagedecoder0": {
                "props": {
                    "cvProcessor": "opencv",
                    "outputDataFormat": "RGB",
                    "dataType": "float32"
                },
                "factory": "mxpi_imagedecoder",
                "next": "mxpi_imageresize0"
            },
            "mxpi_imageresize0": {
                "props": {
                    "cvProcessor": "opencv",
                    "resizeType": "Resizer_Stretch",
                    "dataSource": "mxpi_imagedecoder0",
                    "resizeHeight": "96",
                    "resizeWidth": "96"
                },
                "factory": "mxpi_imageresize",
                "next": "mxpi_imagenormalize0"
            },
            "mxpi_imagenormalize0": {
                "props": {
                    "dataSource": "mxpi_imageresize0",
                    "alpha": "0, 0, 0",
                    "beta": "255, 255, 255",
                    "dataType": "FLOAT32"
                },
                "factory": "mxpi_imagenormalize",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "mxpi_imageresize0",
                    "modelPath": "../models/face_quality_assessment/FQA.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "mxpi_dumpdata0"
            },
            "mxpi_dumpdata0": {
                "props": {
                    "requiredMetaDataKeys": "mxpi_tensorinfer0"
                },
                "factory": "mxpi_dumpdata",
                "next": "appsink0"
            },
            "appsink0": {
                "props": {
                    "blocksize": "4096000"
                },
                "factory": "appsink"
            }
        }
    }

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # 构建流的输入对象--检测目标
    dataInput = MxDataInput()
    if os.path.exists('test.jpg') != 1:
        print("The test image does not exist.")

    with open("test.jpg", 'rb') as f:
        dataInput.data = f.read()

    streamName = b'detection'
    inPluginId = 0
    # 根据流名将检测目标传入流中
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keys = [
        b"appsrc0", b"mxpi_imagedecoder0", b"mxpi_imageresize0", b"mxpi_imagenormalize0", b"mxpi_tensorinfer0"
    ]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    # 从流中取出对应插件的输出数据
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].data.decode()))
        exit()

    print(infer_result)
    # # appsrc0 输出信息
    # visionList = MxpiDataType.MxpiVisionList()
    # visionList.ParseFromString(infer_result[0].messageBuf)
    # vision_data = visionList.visionVec[0].visionData.dataStr
    # visionInfo = visionList.visionVec[0].visionInfo
    # destroy streams
    streamManagerApi.DestroyAllStreams()
