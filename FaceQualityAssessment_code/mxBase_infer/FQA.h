#ifndef UNET_SEGMENTATION_H
#define UNET_SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class FQA {
public:

    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat, int& height, int& width);
    APP_ERROR ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat, MxBase::ResizedImageInfo &resziedImageInfo);
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::string& imgPath, const int height, const int width, std::vector<MxBase::TensorBase> &inputs, 
        std::vector<std::vector<float>>& kp_error_all, std::vector<std::vector<float>>& eulers_error_all, std::vector<float>& kp_ipn);
    APP_ERROR Process(const std::string &testPath);

private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
};

#endif