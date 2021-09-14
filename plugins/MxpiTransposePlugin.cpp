#include "MxpiTransposePlugin.h"
#include "MxBase/Log/Log.h"
#include <iostream>
#include <math.h>

using namespace MxPlugins;
using namespace MxTools;
using namespace std;

namespace {
    const string SAMPLE_KEY = "MxpiVisionList";
}

APP_ERROR MxpiTransposePlugin::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) {
    LogInfo << "MxpiTransposePlugin::Init start.";
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    // parentName_: mxpi_imageresize0
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> descriptionMessageProSptr = std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    return APP_ERR_OK;
}

APP_ERROR MxpiTransposePlugin::DeInit() {
    LogInfo << "MxpiTransposePlugin::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiTransposePlugin::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,const MxpiErrorInfo mxpiErrorInfo) {
    APP_ERROR ret = APP_ERR_OK;
    // Define an object of MxpiMetadataManager
    MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(pluginName, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

APP_ERROR MxpiTransposePlugin::Transpose(MxpiVisionList srcMxpiVisionList, MxpiVisionList& dstMxpiVisionList) {
    dstMxpiVisionList = srcMxpiVisionList;
    for (int i=0;i<dstMxpiVisionList.visionvec_size();i++) {
        MxpiVision* dstMxpiVision = dstMxpiVisionList.mutable_visionvec(i);
        // set MxpiVisionData
        MxpiVisionData* dstMxpiVisionData = dstMxpiVision->mutable_visiondata();
        float HWCData[96*96*3], CHWData[3*96*96];
        std::memcpy(HWCData, (void*)dstMxpiVisionData->dataptr(), sizeof(HWCData));
        for (int c=0;c<3;c++) {
            for (int k=0;k<96*96;k++) CHWData[k+c*96*96] = HWCData[3*k+c];
        }
        std::memcpy((void*)dstMxpiVisionData->dataptr(), CHWData, sizeof(CHWData));
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiTransposePlugin::Process(std::vector<MxpiBuffer*>& mxpiBuffer) {
    LogInfo << "MxpiTransposePlugin::Process start";
    // Get the data from buffer
    MxpiBuffer* buffer = mxpiBuffer[0];
    // Get metadata by key.
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiTransposePlugin process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiTransposePlugin process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the data from buffer(mxpi_imageresize0)
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }
    // check whether the proto struct name is MxpiVisionList(plugin mxpi_imageresize's output format)
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != SAMPLE_KEY) {
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_) << "Proto struct name is not MxpiVisionList, failed";
        mxpiErrorInfo.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }
    // Generate sample output
    shared_ptr<MxpiVisionList> srcMxpiVisionListSptr = static_pointer_cast<MxpiVisionList>(metadata);
    shared_ptr<MxpiVisionList> dstMxpiVisionListSptr = make_shared<MxpiVisionList>();
    APP_ERROR ret = Transpose(*srcMxpiVisionListSptr, *dstMxpiVisionListSptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "MxpiTransposePlugin gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Add Generated data to metedata
    // 通过调用“AddProtoMetadata()”，将结果挂载至获取输入时对应的Buffer
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiVisionListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiTransposePlugin add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiTransposePlugin::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiTransposePlugin::DefineProperties() {
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "dataSource", "name", "the name of previous plugin", "mxpi_imageresize0", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiTransposePlugin)
