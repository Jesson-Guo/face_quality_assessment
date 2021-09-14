#ifndef SDKMEMORY_MxpiTransposePlugin_H
#define SDKMEMORY_MxpiTransposePlugin_H
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
/**
* @api
* @brief Definition of MxpiTransposePlugin class.
*/
namespace MxPlugins {
    class MxpiTransposePlugin : public MxTools::MxPluginBase {
    public:
        /**
         * @api
         * @brief Initialize configure parameter.
         * @param configParamMap
         * @return APP_ERROR
         */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) override;
        /**
         * @api
         * @brief DeInitialize configure parameter.
         * @return APP_ERROR
         */
        APP_ERROR DeInit() override;
        /**
         * @api
         * @brief Process the data of MxpiBuffer.
         * @param mxpiBuffer
         * @return APP_ERROR
         */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer *>& mxpiBuffer) override;
        /**
         * @api
         * @brief Definition the parameter of configure properties.
         * @return std::vector<std::shared_ptr<void>>
         */
        static std::vector<std::shared_ptr<void>> DefineProperties();
        /**
         * @api
         * @brief convert from HWC to CHW.
         * @param key
         * @param buffer
         * @return APP_ERROR
         */
        APP_ERROR transpose(MxTools::MxpiVisionList srcMxpiVisionList, MxTools::MxpiVisionList& dstMxpiVisionList);

    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer& buffer, const std::string pluginName, const MxTools::MxpiErrorInfo mxpiErrorInfo);
        std::string parentName_;
        std::ostringstream ErrorInfo_;
    };
}
#endif //SDKMEMORY_MxpiTransposePlugin_H