```
atc --input_format=NCHW --framework=1 --model=FaceQualityAssessment.air --output=FQA --output_type=FP32 --soc_version=Ascend310 --insert_op_conf=FaceQualityAssessment_aipp.cfg
```

```
python3.7 -u fqa_opencv.py --dataset=/home/cd_mindx/FaceQualityAssessment/dataset/AFLW2000 --pipeline=../pipeline/fqa.pipeline --output=/home/cd_mindx/FaceQualityAssessment/infer_result/AFLW2000 > /home/cd_mindx/FaceQualityAssessment/infer/infer.log 2>&1
```

 

E0906 04:00:51.491937 16837 MxStreamManager.cpp:165] [6005][stream invaldid config] pipeline is not a valid json. Parse json value of stream failed. Error message: (* Line 6, Column 9 Syntax error: Malformed object literal).





E0906 05:59:38.968818 16869 MxsmElement.cpp:638] [6015][element invalid properties] Invalid property: tensorFormat, prop value: 1. Please remove this property from the config file.
E0906 05:59:38.969014 16869 MxsmElement.cpp:709] [6015][element invalid properties] mxpi_tensorinfer has an invalid property.
E0906 05:59:38.969034 16869 MxsmStream.cpp:1555] [6015][element invalid properties] mxpi_tensorinfer0 is an invalid element of mxpi_tensorinfer.
E0906 05:59:38.969050 16869 MxsmStream.cpp:661] [6015][element invalid properties] Creates face_quality_assessment Stream failed.
E0906 05:59:38.969066 16869 MxStreamManagerDptr.cpp:425] [6015][element invalid properties] create stream(face_quality_assessment) failed.









E0906 06:28:12.635419 17006 MxpiImageResize.cpp:425] image cols or rows is bigger than padding size, paddingWidth_=32 paddingHeight_=32 imageRGB.cols is 112 imageRGB.rows is 139









E0906 06:31:44.229269 17044 MxpiTensorInfer.cpp:713] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](27648) does not match model inputTensors[0](110592). Tensor Dtype: TENSOR_DTYPE_UINT8, model Dtype: TENSOR_DTYPE_FLOAT32
E0906 06:31:44.229533 17044 MxpiTensorInfer.cpp:749] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](27648) does not match model inputTensors[0](110592). Tensor Dtype: TENSOR_DTYPE_UINT8, model Dtype: TENSOR_DTYPE_FLOAT32
E0906 06:31:44.229555 17044 MxpiTensorInfer.cpp:555] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](27648) does not match model inputTensors[0](110592). Tensor Dtype: TENSOR_DTYPE_UINT8, model Dtype: TENSOR_DTYPE_FLOAT32
E0906 06:31:44.229576 17044 MxpiTensorInfer.cpp:579] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](27648) does not match model inputTensors[0](110592). Tensor Dtype: TENSOR_DTYPE_UINT8, model Dtype: TENSOR_DTYPE_FLOAT32
E0906 06:31:44.229610 17044 MxpiTensorInfer.cpp:188] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](27648) does not match model inputTensors[0](110592). Tensor Dtype: TENSOR_DTYPE_UINT8, model Dtype: TENSOR_DTYPE_FLOAT32









E0906 06:36:23.710665 17047 MxpiImageDecoder.cpp:61] [mxpi_imagedecoder0][1013][initialize failed] Unknown dataType [FLOAT32].
E0906 06:36:23.710918 17047 MxGstBase.cpp:612] [mxpi_imagedecoder0][1013][initialize failed] Plugin initialize failed.



E0906 06:36:23.717617 17047 MxsmStream.cpp:700] [6003][stream change state fail] Failed to set the state of the Stream, named: face_quality_assessment.
E0906 06:36:23.717662 17047 MxStreamManagerDptr.cpp:425] [6003][stream change state fail] create stream(face_quality_assessment) failed.









```
find /home/data/cd_mindx/mxManufacture-2.0.2/ -type f -name "*" | xargs grep "MX_PLUGIN_GENERATE"
```



```
./build/fqa_opencv /home/cd_mindx/FaceQualityAssessment/dataset/AFLW2000
```







```
eval.py > eval.log 2>&1
```











E0910 03:49:59.989284 52925 MxpiTensorInfer.cpp:713] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](110592) does not match model inputTensors[0](27648). Tensor Dtype: TENSOR_DTYPE_FLOAT32, model Dtype: TENSOR_DTYPE_UINT8
E0910 03:49:59.989580 52925 MxpiTensorInfer.cpp:749] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](110592) does not match model inputTensors[0](27648). Tensor Dtype: TENSOR_DTYPE_FLOAT32, model Dtype: TENSOR_DTYPE_UINT8
E0910 03:49:59.989598 52925 MxpiTensorInfer.cpp:555] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](110592) does not match model inputTensors[0](27648). Tensor Dtype: TENSOR_DTYPE_FLOAT32, model Dtype: TENSOR_DTYPE_UINT8
E0910 03:49:59.989616 52925 MxpiTensorInfer.cpp:579] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](110592) does not match model inputTensors[0](27648). Tensor Dtype: TENSOR_DTYPE_FLOAT32, model Dtype: TENSOR_DTYPE_UINT8
E0910 03:49:59.989637 52925 MxpiTensorInfer.cpp:188] [mxpi_tensorinfer0][100017][The input of the model does not match] The datasize of concated inputTensors[0](110592) does not match model inputTensors[0](27648). Tensor Dtype: TENSOR_DTYPE_FLOAT32, model Dtype: TENSOR_DTYPE_UINT8





python3.7 -u fqa_opencv.py --dataset=/home/cd_mindx/FaceQualityAssessment/dataset/test_dataset --pipeline=../pipeline/fqa.pipeline --output=/home/cd_mindx/FaceQualityAssessment/infer_result/test > /home/cd_mindx/FaceQualityAssessment/infer/infer.log 2>&1



atc --input_format=NCHW --framework=1 --model=FaceQualityAssessment.air --output=FQA --output_type=FP32 --soc_version=Ascend310 --insert_op_conf=fqa.cfg

