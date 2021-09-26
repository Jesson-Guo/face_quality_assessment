# 交付件基本信息

**发布者（Publisher）**：Huawei

**应用领域（Application Domain）**：Aesthetics Assessment

**版本（Version）**：1.1

**修改时间（Modified）**：2020.09.16

**大小（Size）**：16 MB (ckpt)/8 MB (air)/4.3 MB (om)

**框架（Framework）**：MindSpore_1.3.0

**模型格式（Model Format）**：ckpt/air/om

**精度（Precision）**：Mixed/FP16

**处理器（Processor）**：昇腾910/昇腾310

**应用级别（Categories）**：Released

**描述（Description）：**基于MindSpore框架的FaceQualityAssessment人脸质量评估网络模型训练并保存模型，通过ATC工具转换，可在昇腾AI设备上运行

# 概述

## 简述

FaceQualityAssessment模型是一个基于 Resnet12 的人脸质量评估网络。

ResNet（残差神经网络）是由何开明等四位微软研究院中国人提出的。通过使用ResNet单元，成功训练了152层神经网络，在ilsvrc2015中获得冠军。 top 5的错误率为3.57%，参数量低于vggnet，效果非常突出。传统的卷积网络或全连接网络或多或少都会有信息丢失。同时会导致梯度消失或爆炸，从而导致深度网络训练失败。 ResNet 在一定程度上解决了这个问题。通过将输入信息传递到输出，信息的完整性得到保护。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。ResNet的结构可以非常快地加速神经网络的训练，模型的准确率也大大提高。同时，ResNet 非常流行，甚至可以直接用在概念网中。

- 参考论文：

  [Paper](https://arxiv.org/pdf/1512.03385.pdf): Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

- 参考实现：https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceQualityAssessment

通过Git获取对应commit_id的代码方法如下：

```
git clone {repository_url}     # 克隆仓库的代码
cd {repository_name}           # 切换到模型的代码仓目录
git checkout  {branch}         # 切换到对应分支
git reset --hard ｛commit_id｝  # 代码设置到对应的commit_id
cd ｛code_path｝                # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

# 推理

## 模型转换

*（将模型转换成OM模型。提供通过ATC工具将模型转换为OM模型的方法，提供ATC转换命令和AIPP配置文件）*

1. 准备模型文件。

   上传模型训练生成的**.air**模型文件

   上传**convert**目录

2. 模型转换。

   在当前容器目录执行

   ```
   bash air2om.sh FaceQualityAssessment.air FQA
   ```

   其中：
   
   - **FaceQualityAssessment.air**文件为模型训练生成的.air文件，
   
   - **FQA**为.om目标文件名。

## mxBase推理

1. 编译工程。

2. 修改配置文件。

   - 修改CMakeLists.txt文件

     ```cmake
     cmake_minimum_required(VERSION 3.14.0)
     
     project(fqa_opencv)
     
     # 将fqa_opencv修改为你的启动文件名。
     set(TARGET fqa_opencv)
     
     add_definitions(-DENABLE_DVPP_INTERFACE)
     add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
     add_definitions(-Dgoogle=mindxsdk_private)
     add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wall)
     add_link_options(-Wl,-z,relro,-z,now,-z,noexecstack -s -pie)
     
     if(NOT DEFINED ENV{ASCEND_HOME})
         message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
     endif()
     if(NOT DEFINED ENV{ASCEND_VERSION})
         message(WARNING "please define environment variable:ASCEND_VERSION")
     endif()
     if(NOT DEFINED ENV{ARCH_PATTERN})
         message(WARNING "please define environment variable:ARCH_PATTERN")
     endif()
     set(ACL_INC_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/acllib/include)
     set(ACL_LIB_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/acllib/lib64)
     
     set(MXBASE_ROOT_DIR $ENV{MX_SDK_HOME})
     set(MXBASE_INC ${MXBASE_ROOT_DIR}/include)
     set(MXBASE_LIB_DIR ${MXBASE_ROOT_DIR}/lib)
     set(MXBASE_POST_LIB_DIR ${MXBASE_ROOT_DIR}/lib/modelpostprocessors)
     set(MXBASE_POST_PROCESS_DIR ${MXBASE_ROOT_DIR}/include/MxBase/postprocess/include)
     
     if(DEFINED ENV{MXSDK_OPENSOURCE_DIR})
         set(OPENSOURCE_DIR ${ENV{MXSDK_OPENSOURCE_DIR})
     else()
         set(OPENSOURCE_DIR ${MXBASE_ROOT_DIR}/opensource)
     endif()
     
     include_directories(${ACL_INC_DIR})
     include_directories(${OPENSOURCE_DIR}/include)
     include_directories(${OPENSOURCE_DIR}/include/opencv4)
     
     include_directories(${MXBASE_INC})
     include_directories(${MXBASE_POST_PROCESS_DIR})
     
     link_directories(${ACL_LIB_DIR})
     link_directories(${OPENSOURCE_DIR}/lib)
     link_directories(${MXBASE_LIB_DIR})
     link_directories(${MXBASE_POST_LIB_DIR})
     
     # 将FQA.cpp修改为对应的cpp文件。
     add_executable(${TARGET} main.cpp FQA.cpp)
     target_link_libraries(${TARGET} glog cpprest mxbase opencv_world stdc++fs)
     
     install(TARGETS ${TARGET} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)
     ```

3. 运行推理服务。

   解压**FaceQualityAssessment/infer/data/input**目录下的**AFLW2000.zip**数据集到**AFLW2000**

   执行

   ```
   bash build.sh ../data/input/AFLW2000 ../convert/FQA.om
   ```

   其中：

   - **../data/input/AFLW2000**为数据集所在位置，

   - **../convert/FQA.om**为om模型文件所在位置。

4. 观察结果。

   ```
   -- The C compiler identification is GNU 7.5.0
   -- The CXX compiler identification is GNU 7.5.0
   -- Check for working C compiler: /usr/bin/cc
   -- Check for working C compiler: /usr/bin/cc -- works
   ...
   Scanning dependencies of target fqa_opencv
   make[2]: Leaving directory '/home/cd_mindx/FaceQualityAssessment/infer/mxbase/build'
   make[2]: Entering directory '/home/cd_mindx/FaceQualityAssessment/infer/mxbase/build'
   [ 33%] Building CXX object CMakeFiles/fqa_opencv.dir/src/main.cpp.o
   ...
   I0922 13:10:11.389950 34047 ModelInferenceProcessor.cpp:22] Begin to ModelInferenceProcessor init
   I0922 13:10:11.424309 34047 ModelInferenceProcessor.cpp:69] End to ModelInferenceProcessor init
   ========== 5 keypoints average err: [3.30475, 3.7926, 3.62504, 3.12129, 2.78628]
   ========== 3 eulers average err: [21.4818, 15.8031, 17.0047]
   IPN of 5 keypoints: 18.2403
   MAE of elur: 18.0966
   I0922 13:10:20.231670 34047 DeviceManager.cpp:83] DestroyDevices begin
   I0922 13:10:20.231690 34047 DeviceManager.cpp:85] destory device:0
   I0922 13:10:20.396395 34047 DeviceManager.cpp:91] aclrtDestroyContext successfully!
   I0922 13:10:21.320482 34047 DeviceManager.cpp:99] DestroyDevices successfully
   ```

   **推理结果为：**

   ========== 5 keypoints average err: [3.30475, 3.7926, 3.62504, 3.12129, 2.78628]
   ========== 3 eulers average err: [21.4818, 15.8031, 17.0047]
   IPN of 5 keypoints: 18.2403
   MAE of elur: 18.0966

## MindX SDK推理

2. 修改配置文件。

   *（提供pipeline文件和后处理文件参数说明，方便用户修改参数。）*

   1. 修改pipeline文件。

      ```pipeline
      {
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
                  "next": "mxpi_transposeplugin0"
              },
              "mxpi_transposeplugin0": {
                  "props": {
                      "dataSource": "mxpi_imageresize0"
                  },
                  "factory": "mxpi_transposeplugin",
                  "next": "mxpi_imagenormalize0"
              },
              "mxpi_imagenormalize0": {
                  "props": {
                      "dataSource": "mxpi_transposeplugin0",
                      "alpha": "0, 0, 0",
                      "beta": "255, 255, 255",
                      "dataType": "FLOAT32"
                  },
                  "factory": "mxpi_imagenormalize",
                  "next": "mxpi_tensorinfer0"
              },
              "mxpi_tensorinfer0": {
                  "props": {
                      "dataSource": "mxpi_imagenormalize0",
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
      ```

      修改插件**mxpi_tensorinfer0**中的 **"modelPath": "../models/face_quality_assessment/FQA.omom"**，将**modelPath**修改为对应的om文件路径。

      修改插件**mxpi_transposeplugin0**中的 **"dataSource": "mxpi_imageresize0"**，将**mxpi_imageresize0**修改为上游插件名，默认为**mxpi_imageresize0**。

2. 运行推理服务。

   提供运行推理服务中涉及的参数说明。

   1. 进入sdk/目录

   2. 执行推理。

      ```
      bash run.sh ../data/input/AFLW2000 ../data/config/fqa.pipeline
      ```

      其中：

      - **../data/input/AFLW2000**为数据集所在位置，

      - **../data/config/fqa.pipeline**为pipeline文件所在位置。

   3. 查看推理结果。

      ```
      cat infer.log
      ```

      **推理结果为：**

      ========== 5 keypoints average err:['3.311', '3.976', '3.845', '3.156', '2.939']
      ========== 3 eulers average err:['21.482', '15.809', '17.008']
      ========== IPN of 5 keypoints:18.986170137889545
      ========== MAE of elur:18.099490226755503

4. 执行精度测试。

5. 查看性能结果。

   - 打开性能统计开关。将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6。

     ```
     vim /home/HwHiAiUser/mxManufacture-2.0.2/config/sdk.conf执行run.sh脚本。
     ```

   - 执行推理。

   - 在日志目录“/home/HwHiAiUser/mxManufacture-2.0.2/logs/”查看性能统计结果。

     ```
     performance—statistics.log.e2e.xxx
     performance—statistics.log.plugin.xxx
     performance—statistics.log.tpr.xxx
     ```

# 在ModelArts上应用

## 创建OBS桶

1. 创建桶。

   登录[OBS管理控制台](https://storage.huaweicloud.com/obs)，创建OBS桶。具体请参见[创建桶](https://support.huaweicloud.com/usermanual-obs/obs_03_0306.html)章节。

2. 创建文件夹存放数据。

   创建用于存放数据的文件夹，具体请参见[新建文件夹](https://support.huaweicloud.com/usermanual-obs/obs_03_0316.html)章节。

   *目录结构说明：*

   ![image-20210917112951879](C:\Users\Jesson\AppData\Roaming\Typora\typora-user-images\image-20210917112951879.png)
   
   - code：存放训练脚本目录，上传**FaceQualityAssessment**中的代码，上传modelarts中的**start.py**到code目录下
   - dataset：存放训练数据集目录，将**FaceQualityAssessment\infer\data\input**目录下的数据上传到dataset
   - logs：存放训练日志目录
   - model：训练生成ckpt模型以及om模型目录

## 创建训练作业

*描述要点（key）：指导用户在ModelArts上创建训练作业，提供训练参数配置说明*

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“训练管理 > 训练作业”，默认进入“训练作业”列表。

2. 在训练作业列表中，单击左上角“创建”，进入“创建训练作业”页面。

3. 在创建训练作业页面，填写训练作业相关参数，然后单击“下一步”。

   1. 如果没有算法，则提前创建算法，具体请参见[创建算法](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0233.html)章节。启动文件为**start.py**。

      ![image-20210923142216957](C:\Users\Jesson\AppData\Roaming\Typora\typora-user-images\image-20210923142216957.png)

   2. 如果已经有创建好的算法，则选择一个算法，填写相关输入输出路径

      ![image-20210923135019927](C:\Users\Jesson\AppData\Roaming\Typora\typora-user-images\image-20210923135019927.png)

   3. 选择资源类型为**“Ascend”**；节点个数选择**1*Ascend**；选择日志输出路径

      ![image-20210917114005814](C:\Users\Jesson\AppData\Roaming\Typora\typora-user-images\image-20210917114005814.png)

   4. 在“规格确认”页面，确认填写信息无误后，单击“提交”，完成训练作业的创建。

4. 训练作业一般需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟到几十分钟不等。

## 查看训练任务日志

训练完成后，在log文件夹下查看本次训练的日志。

![image-20210917114516252](C:\Users\Jesson\AppData\Roaming\Typora\typora-user-images\image-20210917114516252.png)

训练成功后，在model文件夹下可以查看最终的日志文件。

## 查看冻结模型

训练完成后，在model/output文件夹下查看冻结的模型。

![image-20210923142943779](C:\Users\Jesson\AppData\Roaming\Typora\typora-user-images\image-20210923142943779.png)