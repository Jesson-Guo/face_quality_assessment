#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <dirent.h>
#include <stdio.h>
#include "FQA.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"

using namespace MxBase;

template<class Iter>
inline size_t argmax(Iter first, Iter last) {
    return std::distance(first, std::max_element(first, last));
}

APP_ERROR FQA::Init(const InitParam& initParam) {
    // 设备初始化
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    // 上下文初始化
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    // 加载模型
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR FQA::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR FQA::ReadImage(const std::string& imgPath, cv::Mat& imageMat, int& height, int& width) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    height = imageMat.cols;
    width = imageMat.rows;
    cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
    return APP_ERR_OK;
}

APP_ERROR FQA::ResizeImage(const cv::Mat& srcImageMat, float* transMat) {
    static constexpr uint32_t resizeHeight = 96;
    static constexpr uint32_t resizeWidth = 96;
    cv::Mat dstImageMat;
    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeHeight, resizeWidth));
    // 转成NCHW
    for (int i=0;i<dstImageMat.rows;i++) {
        for (int j=0;j<dstImageMat.cols;j++) {
            transMat[i*resizeHeight+j] = (float)dstImageMat.at<cv::Vec3b>(i, j)[0] / 255;
            transMat[resizeHeight*resizeWidth+i*resizeHeight+j] = (float)dstImageMat.at<cv::Vec3b>(i, j)[1] / 255;
            transMat[2*resizeHeight*resizeWidth+i*resizeHeight+j] = (float)dstImageMat.at<cv::Vec3b>(i, j)[2] / 255;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR FQA::VectorToTensorBase(float* transMat, MxBase::TensorBase& tensorBase) {
    const uint32_t dataSize = 3*96*96*sizeof(float);
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(transMat, dataSize, MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {3, static_cast<uint32_t>(96*96)};
    tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR FQA::Inference(const std::vector<MxBase::TensorBase>& inputs, std::vector<MxBase::TensorBase>& outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR FQA::Process(const std::string& testPath) {
    DIR* directory_pointer = NULL;
    struct dirent* entry;
    if ((directory_pointer=opendir(testPath.c_str()))==NULL) {
        printf("Error open\n");
        exit(0);
    } else {
        std::vector<std::vector<float>> kp_error_all(5), eulers_error_all(3);
        std::vector<float> kp_ipn;
        while ((entry=readdir(directory_pointer))!=NULL) {
            if (entry->d_name[0] == '.') continue;
            std::string s = entry->d_name;
            if (s.substr(s.length()-3, 3) == "jpg") {
                std::string imgPath = testPath[testPath.length()-1]=='/' ? testPath+s : testPath+"/"+s;
                std::string txtPath = imgPath.substr(0, imgPath.length()-3) + "txt";
                
                cv::Mat imageMat;
                int height, width;
                APP_ERROR ret = ReadImage(imgPath, imageMat, height, width);
                if (ret != APP_ERR_OK) {
                    LogError << "ReadImage failed, ret=" << ret << ".";
                    return ret;
                }

                float transMat[27648];
                ResizeImage(imageMat, transMat);

                TensorBase tensorBase;
                ret = VectorToTensorBase(transMat, tensorBase);
                if (ret != APP_ERR_OK) {
                    LogError << "VectorToTensorBase failed, ret=" << ret << ".";
                    return ret;
                }

                std::vector<MxBase::TensorBase> inputs = {};
                std::vector<MxBase::TensorBase> outputs = {};
                inputs.push_back(tensorBase);
                ret = Inference(inputs, outputs);
                if (ret != APP_ERR_OK) {
                    LogError << "Inference failed, ret=" << ret << ".";
                    return ret;
                }

                MxBase::TensorBase& tensor0 = outputs[0];
                ret = tensor0.ToHost();
                if (ret != APP_ERR_OK) {
                    LogError << GetError(ret) << "Tensor_0 deploy to host failed.";
                    return ret;
                }
                auto eulers_ori = reinterpret_cast<float (*)>(tensor0.GetBuffer());
                eulers_ori[0] *= 90;
                eulers_ori[1] *= 90;
                eulers_ori[2] *= 90;

                MxBase::TensorBase& tensor1 = outputs[1];
                ret = tensor1.ToHost();
                if (ret != APP_ERR_OK) {
                    LogError << GetError(ret) << "Tensor_1 deploy to host failed.";
                    return ret;
                }
                auto heatmap = reinterpret_cast<float (*)[48][48]>(tensor1.GetBuffer());
                
                std::ifstream in(txtPath);
                bool euler_kps_do(true);
                float eulgt[3];
                int kp_list[5][2] = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}, kp_coord_ori[5][2];
                if (!in) {
                    euler_kps_do = false;
                    continue;
                }
                else {
                    for (int i=0;i<3;i++) in >> eulgt[i];
                    for (int i=0;i<10;i++) {
                        float x, y;
                        in >> x >> y;
                        if (x<0 || y<0) continue;
                        kp_list[i][0] = (int) ((x*96)/(float)width);
                        kp_list[i][1] = (int) ((y*96)/(float)height);
                    }
                }

                for (int i=0;i<5;i++) {
                    std::vector<float> map_1(48*48, 0);
                    float soft_sum(0);
                    int o=0;
                    for (int j=0;j<48;j++) {
                        for (int k=0;k<48;k++) {
                            map_1[o] = exp(heatmap[i][j][k]);
                            o++;
                            soft_sum += exp(heatmap[i][j][k]);
                        }
                    }
                    for (int j=0;j<48*48;j++) map_1[j] = map_1[j] / soft_sum;
                    int kp_coor = static_cast<int>(argmax(map_1.begin(), map_1.end()));
                    kp_coord_ori[i][0] = (kp_coor % 48) * 2;
                    kp_coord_ori[i][1] = (kp_coor / 48) * 2;
                }

                if (euler_kps_do) {
                    for (int i=0;i<3;i++) eulers_error_all[i].push_back(abs(eulers_ori[i] - eulgt[i]));
                    bool cur_flag = true;
                    float eye_dis(1.0);
                    if (kp_list[0][0] < 0 || kp_list[0][1] < 0 || kp_list[1][0] < 0 || kp_list[1][1] < 0) cur_flag = false;
                    else eye_dis = sqrt(pow(abs(kp_list[0][0] - kp_list[1][0]), 2) + pow(abs(kp_list[0][1] - kp_list[1][1]), 2));
                    float cur_error_sum(0), cnt(0);
                    for (int i=0;i<5;i++) {
                        if (kp_list[i][0] != -1) {
                            float dis = sqrt(pow(kp_list[i][0] - kp_coord_ori[i][0], 2) + pow(kp_list[i][1] - kp_coord_ori[i][1], 2));
                            kp_error_all[i].push_back(dis);
                            cur_error_sum += dis;
                            cnt++;
                        }
                    }
                    if (cur_flag) kp_ipn.push_back(cur_error_sum / cnt / eye_dis);
                }
            }
        }

        std::vector<float> kp_ave_error, euler_ave_error;
        for (uint32_t i=0;i<kp_error_all.size();i++) {
            float kp_error_all_sum(0);
            for (uint32_t j=0;j<kp_error_all[i].size();j++) kp_error_all_sum += kp_error_all[i][j];
            kp_ave_error.push_back(kp_error_all_sum/kp_error_all[i].size());
        }
        for (uint32_t i=0;i<eulers_error_all.size();i++) {
            float euler_ave_error_sum(0);
            for (uint32_t j=0;j<eulers_error_all[i].size();j++) euler_ave_error_sum += eulers_error_all[i][j];
            euler_ave_error.push_back(euler_ave_error_sum/eulers_error_all[i].size());
        }
        std::cout << "========== 5 keypoints average err: [" << kp_ave_error[0];
        for (uint32_t i=1;i<kp_ave_error.size();i++) std::cout << ", " << kp_ave_error[i];
        std::cout << "]\n";

        std::cout << "========== 3 eulers average err: [" << euler_ave_error[0];
        for (uint32_t i=1;i<euler_ave_error.size();i++) std::cout << ", " << euler_ave_error[i];
        std::cout << "]\n";

        float ipn(0), mae(0);
        for (uint32_t i=0;i<kp_ipn.size();i++) ipn += kp_ipn[i];
        std::cout << "IPN of 5 keypoints: " << (ipn / kp_ipn.size()) * 100 << "\n";

        for (uint32_t i=0;i<euler_ave_error.size();i++) mae += euler_ave_error[i];
        std::cout << "MAE of elur: " << mae / euler_ave_error.size() << "\n";
    }
    closedir(directory_pointer);

    return APP_ERR_OK;
}
