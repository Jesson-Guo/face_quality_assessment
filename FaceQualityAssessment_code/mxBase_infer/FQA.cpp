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

APP_ERROR FQA::ResizeImage(const cv::Mat& srcImageMat, cv::Mat& dstImageMat, MxBase::ResizedImageInfo& resizedImageInfo) {
    static constexpr uint32_t resizeHeight = 96;
    static constexpr uint32_t resizeWidth = 96;

    resizedImageInfo.heightOriginal = srcImageMat.rows;
    resizedImageInfo.heightResize = resizeHeight;
    resizedImageInfo.widthOriginal = srcImageMat.cols;
    resizedImageInfo.widthResize = resizeWidth;
    resizedImageInfo.resizeType = RESIZER_STRETCHING;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeHeight, resizeWidth));
    // 可能不需要转成NCHW？
    return APP_ERR_OK;
}

APP_ERROR FQA::CVMatToTensorBase(const cv::Mat& imageMat, MxBase::TensorBase& tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * YUV444_RGB_WIDTH_NU;
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(imageMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {imageMat.rows * YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
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

APP_ERROR FQA::PostProcess(const std::string& imgPath, const int height, const int width, std::vector<MxBase::TensorBase>& inputs,
    std::vector<std::vector<float>>& kp_error_all, std::vector<std::vector<float>>& eulers_error_all, std::vector<float>& kp_ipn) {
    MxBase::TensorBase& tensor = inputs[1];
    int ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor_1 deploy to host failed.";
        return ret;
    }
    auto heatmap = reinterpret_cast<float (*)[48][48]>(tensor.GetBuffer());

    tensor = inputs[0];
    ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor_0 deploy to host failed.";
        return ret;
    }
    auto eulers_ori = reinterpret_cast<float (*)>(tensor.GetBuffer());

    std::string txtPath = imgPath;
    while (txtPath.substr(txtPath.length()-3,3) == "jpg")
        txtPath.replace(txtPath.length()-3, 3, "txt");
    // read ground truth
    std::ifstream in;
    float eulgt[3];
    int kp_list[5][2], kp_coord_ori[5][2];
    in.open(txtPath);
    for (auto i=0;i<3;++i) in >> eulgt[i];
    for (auto i=0;i<10;++i) {
        float x;
        in >> x;
        if (i%2==0) kp_list[(i-3)/2][0] = int(x/width*96);
        else kp_list[(i-3)/2][1] = int(x/height*96);
    }
    // get infer coordinates
    for (auto i=0;i<5;i++) {
        std::vector<float> map_1(48*48, 0);
        float soft_sum(0);
        int o=0;
        for (auto j=0;j<48;j++) {
            for (auto k=0;k<48;k++) {
                map_1[o] = exp(heatmap[i][j][k]);
                o++;
                soft_sum += exp(heatmap[i][j][k]);
            }
        }
        for (auto j=0;j<48*48;j++) map_1[j] = map_1[j] / soft_sum;
        int kp_coor = static_cast<int>(argmax(map_1.begin(), map_1.end()));
        kp_coord_ori[i][0] = int((kp_coor % 48) * 2.0);
        kp_coord_ori[i][1] = int((kp_coor / 48) * 2.0);
    }
    for (auto i=0;i<3;i++) eulers_error_all[i].push_back(abs(eulers_ori[i] - eulgt[i]));
    float eye_dis = sqrt(pow(abs(kp_list[0][0] - kp_list[1][0]), 2) + pow(abs(kp_list[0][1] - kp_list[1][1]), 2));
    std::vector<float> cur_error_list;
    float cur_error_sum(0);
    for (auto i=0;i<5;i++) {
        if (kp_list[i][0] != -1) {
            float dis = sqrt(pow(kp_list[i][0] - kp_coord_ori[i][0], 2) + pow(kp_list[i][1] - kp_coord_ori[i][1], 2));
            kp_error_all[i].push_back(dis);
            cur_error_sum += dis;
        }
    }
    kp_ipn.push_back(cur_error_sum / 5 / eye_dis);
    return APP_ERR_OK;
}

APP_ERROR FQA::Process(const std::string& testPath) {
    DIR* directory_pointer = NULL;
    /*struct dirent {
        long d_ino; // inode number 索引节点号
        off_t d_off; // offset to this dirent 在目录文件中的偏移
        unsigned short d_reclen; // length of this d_name 文件名长
        unsigned char d_type; // the type of d_name 文件类型
        char d_name [NAME_MAX+1]; // file name (null-terminated) 文件名，最长255字符
    }
    */
    struct dirent* entry;
    if ((directory_pointer=opendir(testPath.c_str()))==NULL) {
        printf("Error open\n");
        exit(0);
    } else {
        std::vector<std::vector<float>> kp_error_all(5), eulers_error_all(3); // [[], [], [], [], []]     [[], [], []]
        std::vector<float> kp_ipn, kp_ave_error, euler_ave_error; // []
        while ((entry=readdir(directory_pointer))!=NULL) {
            if (entry->d_name[0] == '.') continue;
            std::string s = entry->d_name;
            if (s.substr(s.length()-3, 3) == "jpg") {
                std::string imgPath = testPath[testPath.length()-1]=='/' ? testPath+s : testPath+"/"+s;
                cv::Mat imageMat;
                int height, width;
                APP_ERROR ret = ReadImage(imgPath, imageMat, height, width);
                if (ret != APP_ERR_OK) {
                    LogError << "ReadImage failed, ret=" << ret << ".";
                    return ret;
                }

                ResizedImageInfo resizedImageInfo;
                ResizeImage(imageMat, imageMat, resizedImageInfo);

                TensorBase tensorBase;
                ret = CVMatToTensorBase(imageMat, tensorBase);
                if (ret != APP_ERR_OK) {
                    LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
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
                ret = PostProcess(imgPath, height, width, outputs, kp_error_all, eulers_error_all, kp_ipn);
                if (ret != APP_ERR_OK) {
                    LogError << "PostProcess failed, ret=" << ret << ".";
                    return ret;
                }
            }
        }
        for (auto i=0;i<5;i++) {
            float kp_error_all_sum(0);
            for (auto j=0;j<kp_error_all[i].size();j++) kp_error_all_sum += kp_error_all[i][j];
            kp_ave_error.push_back(kp_error_all_sum/kp_error_all[i].size());
        }
        for (auto i=0;i<3;i++) {
            float euler_ave_error_sum(0);
            for (auto j=0;j<eulers_error_all[i].size();j++) euler_ave_error_sum += eulers_error_all[i][j];
            euler_ave_error.push_back(euler_ave_error_sum/eulers_error_all[i].size());
        }
        std::cout << "========== 5 keypoints average err: [" << kp_ave_error[0];
        for (auto i=1;i<kp_ave_error.size();i++) std::cout << ", " << kp_ave_error[i];
        std::cout << "]\n";

        std::cout << "========== 3 eulers average err: [" << euler_ave_error[0];
        for (auto i=1;i<euler_ave_error.size();i++) std::cout << ", " << euler_ave_error[i];
        std::cout << "]\n";

        float ipn(0), mae(0);
        for (auto i=1;i<kp_ipn.size();i++) ipn += kp_ipn[i];
        std::cout << "IPN of 5 keypoints: " << ipn / kp_ipn.size() * 100 << "\n";

        for (auto i=1;i<euler_ave_error.size();i++) mae += euler_ave_error[i];
        std::cout << "IPN of 5 keypoints: " << mae / euler_ave_error.size() << "\n";
    }
    closedir(directory_pointer);

    return APP_ERR_OK;
}
