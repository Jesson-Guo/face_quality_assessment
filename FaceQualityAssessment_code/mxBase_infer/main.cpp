#include "FQA.h"
#include "MxBase/Log/Log.h"

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input test dataset path";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "/home/cd_mindx/FaceQualityAssessment/dataset/AFLW2000/FQA.om";
    FQA fqa;
    APP_ERROR ret = fqa.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FQA init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    ret = fqa.Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "FQA process failed, ret=" << ret << ".";
        fqa.DeInit();
        return ret;
    }

    fqa.DeInit();
    return APP_ERR_OK;
}
