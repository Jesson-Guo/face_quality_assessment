模型交付件模板-众智

- [交付件基本信息](#交付件基本信息.md)
- [概述](#概述.md)
    - [简述](#简述.md)
    - [默认配置](#默认配置.md)
    - [支持特性](#支持特性.md)
- [准备工作](#准备工作.md)
    - [推理环境准备](#推理环境准备.md)
    - [源码介绍](#源码介绍.md)
- [推理](#推理.md)
    - [准备推理数据](#准备推理数据.md)
    - [模型转换](#模型转换.md)
    - [mxBase推理](#mxBase推理.md)
    - [MindX SDK推理](#MindX-SDK推理.md)
- [在ModelArts上应用](#在ModelArts上应用.md)
    - [上传自定义镜像（适用于PyTorch）](#上传自定义镜像（适用于PyTorch）.md)
    - [创建OBS桶](#创建OBS桶.md)
    - [创建算法（适用于MindSpore和TensorFlow）](#创建算法（适用于MindSpore和TensorFlow）.md)
    - [创建训练作业](#创建训练作业.md)
    - [查看训练任务日志](#查看训练任务日志.md)
    - [迁移学习](#迁移学习.md)

## 交付件基本信息

**发布者（Publisher）**：Huawei

**应用领域（Application Domain）**：Aesthetics Assessment

**版本（Version）**：1.1

**修改时间（Modified）**：2021.09.17

**大小（Size）**：16 MB (ckpt)/8 MB (air)/4.3 MB (om)

**框架（Framework）**：MindSpore_1.3.0

**模型格式（Model Format）**：ckpt/air/om

**精度（Precision）**：Mixed/FP16

**处理器（Processor）**：昇腾910/昇腾310

**应用级别（Categories）**：Released

**描述（Description）**：基于MindSpore框架的FaceQualityAssessment人脸质量评估网络模型训练并保存模型，通过ATC工具转换，可在昇腾AI设备上运行

## 概述

- **[简述](#简述.md)**  

- **[默认配置](#默认配置.md)**  

- **[支持特性](#支持特性.md)**  

### 简述

FaceQualityAssessment模型是一个基于 Resnet12 的人脸质量评估网络。

ResNet（残差神经网络）是由何开明等四位微软研究院中国人提出的。通过使用ResNet单元，成功训练了152层神经网络，在ilsvrc2015中获得冠军。 top 5的错误率为3.57%，参数量低于vggnet，效果非常突出。传统的卷积网络或全连接网络或多或少都会有信息丢失。同时会导致梯度消失或爆炸，从而导致深度网络训练失败。 ResNet 在一定程度上解决了这个问题。通过将输入信息传递到输出，信息的完整性得到保护。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。ResNet的结构可以非常快地加速神经网络的训练，模型的准确率也大大提高。同时，ResNet 非常流行，甚至可以直接用在概念网中。

- 参考论文：

  [Paper](https://arxiv.org/pdf/1512.03385.pdf): Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

- 参考实现：

   https://gitee.com/mindspore/models/tree/master/research/cv/FaceQualityAssessment

- 通过Git获取对应commit_id的代码方法如下：

   ```sh
   git clone {repository_url}     # 克隆仓库的代码
   cd {repository_name}           # 切换到模型的代码仓目录
   git checkout  {branch}         # 切换到对应分支
   git reset --hard ｛commit_id｝  # 代码设置到对应的commit_id
   cd ｛code_path｝                # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
   ```

### 默认配置

- 训练数据集预处理 （以300W-LP训练集为例，仅作为用户参考示例。）

    调整输入图像大小的为（96，96）

    将输入的图像设置为RGB模型

    将图像格式调整为NCHW

    对图像矩阵进行归一化处理

- 测试数据集预处理 （以AFLW2000验证集为例，仅作为用户参考示例。）

    调整输入图像大小的为（96，96）

    将输入的图像设置为RGB模型

    将图像格式调整为NCHW

    对图像矩阵进行归一化处理

- 训练超参
    - Batch size: 256（单卡为32，8卡为256）
    - Momentum: 0.9
    - workers 8
    - Learning rate(LR): 0.02
    - lr_scale: 1
    - Weight decay: 0.0001
    - train epoch: 40

### 支持特性

支持的特性包括：1、分布式训练。2、混合精度。3、数据并行。

8P训练脚本，支持数据并行的分布式训练。脚本样例中默认开启了混合精度。

## 准备工作

- **[推理环境准备](#推理环境准备.md)**  

- **[源码介绍](#源码介绍.md)**  

### 推理环境准备

- 硬件环境、开发环境和运行环境准备请参见[《CANN 软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-upgrade)》。
- 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/home)获取镜像。

    当前模型支持的镜像列表如下表所示。

    **表 1**  镜像列表

    <a name="zh-cn_topic_0000001205858411_table1519011227314"></a>

    <table><thead align="left"><tr id="zh-cn_topic_0000001205858411_row0190152218319"><th class="cellrowborder" valign="top" width="55.00000000000001%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001205858411_p1419132211315"><a name="zh-cn_topic_0000001205858411_p1419132211315"></a><a name="zh-cn_topic_0000001205858411_p1419132211315"></a>镜像名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="20%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001205858411_p75071327115313"><a name="zh-cn_topic_0000001205858411_p75071327115313"></a><a name="zh-cn_topic_0000001205858411_p75071327115313"></a>镜像版本</p>
    </th>
    <th class="cellrowborder" valign="top" width="25%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001205858411_p1024411406234"><a name="zh-cn_topic_0000001205858411_p1024411406234"></a><a name="zh-cn_topic_0000001205858411_p1024411406234"></a>配套CANN版本</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001205858411_row71915221134"><td class="cellrowborder" valign="top" width="55.00000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001205858411_p58911145153514"><a name="zh-cn_topic_0000001205858411_p58911145153514"></a><a name="zh-cn_topic_0000001205858411_p58911145153514"></a>ARM/x86架构：<a href="https://ascendhub.huawei.com/#/detail/infer-modelzoo" target="_blank" rel="noopener noreferrer">infer-modelzoo</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001205858411_p14648161414516"><a name="zh-cn_topic_0000001205858411_p14648161414516"></a><a name="zh-cn_topic_0000001205858411_p14648161414516"></a>21.0.2</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001205858411_p1264815147514"><a name="zh-cn_topic_0000001205858411_p1264815147514"></a><a name="zh-cn_topic_0000001205858411_p1264815147514"></a><a href="https://www.hiascend.com/software/cann/commercial" target="_blank" rel="noopener noreferrer">5.0.2</a></p>
    </td>
    </tr>
    </tbody>
    </table>

### 源码介绍

```txt
/home/HwHiAiUser/vgg16_for_mindspore_{version}_code
├── infer
│   └── README.md
│   ├── convert
│   │   ├──air2om.sh
│   ├── data
│   │   ├── input
│   │   |   ├──AFLW2000.zip
│   │   │   └──training.zip
│   │   ├── model
│   │   │   └──FaceQualityAssessment.air
│   │   └── config
│   │   │   └──fqa.pipeline
│   ├── mxbase
│   │   ├── src
│   │   │   ├── FQA.cpp
│   │   │   ├── FQA.h
│   │   │   └── main.cpp
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk
│   │   ├── fqa_opencv.py
│   │   └── run.sh
│   └── util
│   │   └──plugins
│   │   |   ├── MxpiTransposePlugin.cpp
│   │   |   ├── MxpiTransposePlugin.h
│   │   │   └── CMakeLists.txt
```

## 推理

- **[准备推理数据](#准备推理数据.md)**  

- **[模型转换](#模型转换.md)**  

- **[mxBase推理](#mxBase推理.md)**  

- **[MindX SDK推理](#MindX-SDK推理.md)**  

### 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. 下载源码包。

    单击“下载模型脚本”和“下载模型”，下载所需软件包。

2. 将源码上传至推理服务器任意目录并解压（如：“/home/HwHiAiUser“）。
3. 编译镜像 **（需要安装软件依赖时选择）** 。

    **docker build -t** _infer\_image_ **--build-arg FROM\_IMAGE\_NAME=**_base\_image:tag_ **.**

    **表 1**  参数说明

    <a name="table82851171646"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0304403934_row9243114772414"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0304403934_p524364716241"><a name="zh-cn_topic_0304403934_p524364716241"></a><a name="zh-cn_topic_0304403934_p524364716241"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0304403934_p172431247182412"><a name="zh-cn_topic_0304403934_p172431247182412"></a><a name="zh-cn_topic_0304403934_p172431247182412"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0304403934_row52431473244"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p144312172333"><a name="p144312172333"></a><a name="p144312172333"></a><em id="i290520133315"><a name="i290520133315"></a><a name="i290520133315"></a>infer_image</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0304403934_p10243144712410"><a name="zh-cn_topic_0304403934_p10243144712410"></a><a name="zh-cn_topic_0304403934_p10243144712410"></a>推理镜像名称，根据实际写入。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0304403934_row1624394732415"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0304403934_p92434478242"><a name="zh-cn_topic_0304403934_p92434478242"></a><a name="zh-cn_topic_0304403934_p92434478242"></a><em id="i78645182347"><a name="i78645182347"></a><a name="i78645182347"></a>base_image</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0304403934_p324313472240"><a name="zh-cn_topic_0304403934_p324313472240"></a><a name="zh-cn_topic_0304403934_p324313472240"></a>基础镜像，可从Ascend Hub上下载。</p>
    </td>
    </tr>
    <tr id="row2523459163416"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p55241359203412"><a name="p55241359203412"></a><a name="p55241359203412"></a><em id="i194517711355"><a name="i194517711355"></a><a name="i194517711355"></a>tag</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1952435919341"><a name="p1952435919341"></a><a name="p1952435919341"></a>镜像tag，请根据实际配置，如：21.0.2。</p>
    </td>
    </tr>
    </tbody>
    </table>

4. 准备数据。

   （_准备用于推理的图片、数据集、模型文件、代码等，放在同一数据路径中，如：“/home/HwHiAiUser“。_）

   示例：

   由于后续推理均在容器中进行，因此需要把用于推理的图片、数据集、模型文件、代码等均放在同一数据路径中，后续示例将以“/home/HwHiAiUser“为例。

   ```txt
   ...
   ├── infer
   │   └── README.md
   │   ├── convert
   │   │   ├──air2om.sh
   │   ├── data
   │   │   ├── input
   │   │   |   ├──AFLW2000.zip
   │   │   │   └──training.zip
   │   │   ├── model
   │   │   │   └──FaceQualityAssessment.air
   │   │   └── config
   │   │   │   └──fqa.pipeline
   │   ├── mxbase
   │   └── sdk
   │   └── util
   ```

   AIR模型可通过“模型训练”后转换生成或通过“下载模型”获取。

5. 启动容器。

    进入“infer“目录，执行以下命令，启动容器。

    **bash docker\_start\_infer.sh** _docker\_image:tag_ _model\_dir_

    **表 2**  参数说明

    <a name="table8122633182517"></a>
    <table><thead align="left"><tr id="row16122113320259"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p16122163382512"><a name="p16122163382512"></a><a name="p16122163382512"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p8122103342518"><a name="p8122103342518"></a><a name="p8122103342518"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row11225332251"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p712210339252"><a name="p712210339252"></a><a name="p712210339252"></a><em id="i121225338257"><a name="i121225338257"></a><a name="i121225338257"></a>docker_image</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p0122733152514"><a name="p0122733152514"></a><a name="p0122733152514"></a>推理镜像名称，根据实际写入。</p>
    </td>
    </tr>
    <tr id="row052611279127"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2526192714127"><a name="p2526192714127"></a><a name="p2526192714127"></a><em id="i12120733191212"><a name="i12120733191212"></a><a name="i12120733191212"></a>tag</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16526142731219"><a name="p16526142731219"></a><a name="p16526142731219"></a>镜像tag，请根据实际配置，如：21.0.2。</p>
    </td>
    </tr>
    <tr id="row5835194195611"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p59018537424"><a name="p59018537424"></a><a name="p59018537424"></a>model_dir</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1390135374214"><a name="p1390135374214"></a><a name="p1390135374214"></a>推理代码路径。</p>
    </td>
    </tr>
    </tbody>
    </table>

    启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过修改**docker\_start\_infer.sh**的device来指定挂载的推理芯片。

    ```
    docker run -it \
      --device=/dev/davinci0 \        # 可根据需要修改挂载的npu设备
      --device=/dev/davinci_manager \
    ```

>MindX SDK开发套件（mxManufacture）已安装在基础镜像中，安装路径：“/usr/local/sdk\_home“。

### 模型转换

1. 准备模型文件。

    上传模型训练生成的.air模型文件**或者**使用model目录下的.air文件

2. 模型转换。

    进入“infer/convert“目录进行模型转换，转换详细信息可查看转换脚本和对应的aipp配置文件，**在air2om.sh**脚本文件中，配置相关参数。

    ```
    model_path=$1
    output_model_name=$2

    atc \
    --input_format=NCHW \
    --framework=1 \
    --model=$model_path \
    --output=$output_model_name \
    --output_type=FP32 \
    --soc_version=Ascend310
    ```

    转换命令如下。

    **bash air2om.sh** *model_path output_model_name*

    **表 1**  参数说明

    <a name="table15982121511203"></a>
    <table><thead align="left"><tr id="row1598241522017"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p189821115192014"><a name="p189821115192014"></a><a name="p189821115192014"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1982161512206"><a name="p1982161512206"></a><a name="p1982161512206"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row0982101592015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1598231542020"><a name="p1598231542020"></a><a name="p1598231542020"></a>model_path</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p598231511200"><a name="p598231511200"></a><a name="p598231511200"></a>AIR文件路径。</p>
    </td>
    </tr>
    <tr id="row109831315132011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p598319158204"><a name="p598319158204"></a><a name="p598319158204"></a>output_model_name</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1898316155207"><a name="p1898316155207"></a><a name="p1898316155207"></a>生成的OM文件名，转换脚本会在此基础上添加.om后缀。</p>
    </td>
    </tr>
    </tbody>
    </table>

### mxBase推理

在容器内用mxBase进行推理。

1. 修改配置文件。

   修改CMakeLists.txt文件

     ```
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

2. 编译工程。

   执行：

   **cmake** -S . -B build
   **make** -C ./build -j

3. 运行推理服务。

   执行

   **bash build.sh** *dataset_path om_path*

   **表 2**  参数说明

   <table><thead align="left"><tr id="row1598241522017"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p189821115192014"><a name="p189821115192014"></a><a name="p189821115192014"></a>参数</p>
   </th>
   <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1982161512206"><a name="p1982161512206"></a><a name="p1982161512206"></a>说明</p>
   </th>
   </tr>
   </thead>
   <tbody><tr id="row0982101592015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1598231542020"><a name="p1598231542020"></a><a name="p1598231542020"></a>dataset_path</p>
   </td>
   <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p598231511200"><a name="p598231511200"></a><a name="p598231511200"></a>数据集路径。</p>
   </td>
   </tr>
   <tr id="row109831315132011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p598319158204"><a name="p598319158204"></a><a name="p598319158204"></a>om_path</p>
   </td>
   <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1898316155207"><a name="p1898316155207"></a><a name="p1898316155207"></a>OM文件路径。</p>
   </td>
   </tr>
   </tbody>
   </table>

4. 观察结果。

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
   ...

   **推理结果为：**

   ========== 5 keypoints average err: [3.30475, 3.7926, 3.62504, 3.12129, 2.78628]
   ========== 3 eulers average err: [21.4818, 15.8031, 17.0047]
   IPN of 5 keypoints: 18.2403
   MAE of elur: 18.0966

### MindX SDK推理

1. 修改配置文件。

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
                         "modelPath": "../convert/FQA.om"
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

      其中：

      修改插件**mxpi_tensorinfer0**中的 **"modelPath"**为对应的om文件路径。

      **mxpi_transposeplugin0**为自己编译生成的插件，修改插件**mxpi_transposeplugin0**中的 **"dataSource" **为上游插件名，默认为**mxpi_imageresize0**。

2. 运行推理服务。

   1. 执行推理。

      进入sdk/目录，执行推理：

      **bash run.sh** dataset_path pipeline_path

      **表 3** 参数说明

      <table><thead align="left"><tr id="row1598241522017"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p189821115192014"><a name="p189821115192014"></a><a name="p189821115192014"></a>参数</p>
      </th>
      <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1982161512206"><a name="p1982161512206"></a><a name="p1982161512206"></a>说明</p>
      </th>
      </tr>
      </thead>
      <tbody><tr id="row0982101592015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1598231542020"><a name="p1598231542020"></a><a name="p1598231542020"></a>dataset_path</p>
      </td>
      <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p598231511200"><a name="p598231511200"></a><a name="p598231511200"></a>数据集路径。</p>
      </td>
      </tr>
      <tr id="row109831315132011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p598319158204"><a name="p598319158204"></a><a name="p598319158204"></a>pipeline_path</p>
      </td>
      <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1898316155207"><a name="p1898316155207"></a><a name="p1898316155207"></a>pipeline文件路径。</p>
      </td>
      </tr>
      </tbody>
      </table>

   2. 查看推理结果。

      在当前目录执行：

      **cat infer.log**

      推理结果为：

      ========== 5 keypoints average err:['3.311', '3.976', '3.845', '3.156', '2.939']
      ========== 3 eulers average err:['21.482', '15.809', '17.008']
      ========== IPN of 5 keypoints:18.986170137889545
      ========== MAE of elur:18.099490226755503

3. 执行精度和性能测试。

4. 查看精度和性能结果。

   1. 打开性能统计开关。将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6

   2. 执行推理。

   3. 在日志目录“/home/HwHiAiUser/mxManufacture-2.0.2/logs/”查看性能统计结果。

     performance—statistics.log.e2e.xxx
     performance—statistics.log.plugin.xxx
     performance—statistics.log.tpr.xxx

## 在ModelArts上应用

- **[上传自定义镜像（适用于PyTorch）](#上传自定义镜像（适用于PyTorch）.md)**  

- **[创建OBS桶](#创建OBS桶.md)**  

- **[创建算法（适用于MindSpore和TensorFlow）](#创建算法（适用于MindSpore和TensorFlow）.md)**  

- **[创建训练作业](#创建训练作业.md)**  

- **[查看训练任务日志](#查看训练任务日志.md)**  

- **[迁移学习](#迁移学习.md)**  

### 上传自定义镜像（适用于PyTorch）

1. 从昇腾镜像仓库获取自定义镜像[ascend-pytorch-arm-modelarts](https://ascendhub.huawei.com/#/detail/ascend-pytorch-arm-modelarts)。

2. （可选）如果缺少其他依赖，请在自定义镜像中安装。

   *将需要的依赖体现说明。*

3. 登录[SWR控制台](https://console.huaweicloud.com/swr/?agencyId=5b5810ebce86453a8f77ded5695374cd%C2%AEion%3Dcn-north-4&locale=zh-cn&region=cn-north-4#/app/dashboard)，上传PyTorch训练镜像。具体请参见[容器引擎客户端上传镜像](https://support.huaweicloud.com/qs-swr/)章节。自定义镜像中需要安装tensorboardX。

>SWR的区域需要与ModelArts所在的区域一致。例如：当前ModelArts在华北-北京四区域。SWR所在区域，请选择华北-北京四。

### 创建OBS桶

1. 创建桶。

    登录[OBS管理控制台](https://storage.huaweicloud.com/obs)，创建OBS桶。具体请参见[创建桶](https://support.huaweicloud.com/usermanual-obs/obs_03_0306.html)章节。

2. 创建文件夹存放数据。

    创建用于存放数据的文件夹，具体请参见[新建文件夹](https://support.huaweicloud.com/usermanual-obs/obs_03_0316.html)章节。

    目录结构说明：

    ![](images/0.png)

    目录结构示例：

    - code：存放训练脚本目录
    - dataset：存放训练数据集目录，解压**input/training.zip**后上传
    - logs：存放训练日志目录
    - model：训练生成ckpt模型以及om模型目录

### 创建算法（适用于MindSpore和TensorFlow）

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“算法管理”。
2. 在“我的算法管理”界面，单击左上角“创建”，进入“创建算法”页面。
3. 在“创建算法”页面，填写相关参数，然后单击“提交”。
    1. 设置算法基本信息。

    2. 设置“创建方式”为“自定义脚本”。

       用户需根据实际算法代码情况设置“AI引擎”、“代码目录”和“启动文件”。选择的AI引擎和编写算法代码时选择的框架必须一致。例如编写算法代码使用的是MindSpore，则在创建算法时也要选择MindSpore。

       ![输入图片说明](images/1.png)

       _示例：_

       **表 4** 参数说明

       <a name="table09972489125"></a>
       <table><thead align="left"><tr id="row139978484125"><th class="cellrowborder" valign="top" width="29.470000000000002%" id="mcps1.2.3.1.1"><p id="p16997114831219"><a name="p16997114831219"></a><a name="p16997114831219"></a><em id="i1199720484127"><a name="i1199720484127"></a><a name="i1199720484127"></a>参数名称</em></p>
       </th>
       <th class="cellrowborder" valign="top" width="70.53%" id="mcps1.2.3.1.2"><p id="p199976489122"><a name="p199976489122"></a><a name="p199976489122"></a><em id="i9997154816124"><a name="i9997154816124"></a><a name="i9997154816124"></a>说明</em></p>
       </th>
       </tr>
       </thead>
       <tbody><tr id="row11997124871210"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p1299734820121"><a name="p1299734820121"></a><a name="p1299734820121"></a><em id="i199764819121"><a name="i199764819121"></a><a name="i199764819121"></a>AI引擎</em></p>
       </td>
       <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p1899720481122"><a name="p1899720481122"></a><a name="p1899720481122"></a><em id="i9997848191217"><a name="i9997848191217"></a><a name="i9997848191217"></a>Ascend-Powered-Engine，mindspore_1.3.0-cann_5.0.2</em></p>
       </td>
       </tr>
       <tr id="row5997348121218"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p139971748141218"><a name="p139971748141218"></a><a name="p139971748141218"></a><em id="i1199784811220"><a name="i1199784811220"></a><a name="i1199784811220"></a>代码目录</em></p>
       </td>
       <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p2099724810127"><a name="p2099724810127"></a><a name="p2099724810127"></a><em id="i17997144871212"><a name="i17997144871212"></a><a name="i17997144871212"></a>算法代码存储的OBS路径。上传训练脚本，如：/obs桶/mindspore-dataset/code/cspdarknet53</em></p>
       </td>
       </tr>
       <tr id="row899794811124"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p799714482129"><a name="p799714482129"></a><a name="p799714482129"></a><em id="i399704871210"><a name="i399704871210"></a><a name="i399704871210"></a>启动文件</em></p>
       </td>
       <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p13997154831215"><a name="p13997154831215"></a><a name="p13997154831215"></a><em id="i11997648161214"><a name="i11997648161214"></a><a name="i11997648161214"></a>启动文件：启动训练的python脚本，如：/obs桶/mindspore-dataset/code/cspdarknet53/start.py</em></p>
       <div class="notice" id="note1799734891214"><a name="note1799734891214"></a><a name="note1799734891214"></a><span class="noticetitle"> 须知： </span><div class="noticebody"><p id="p7998194814127"><a name="p7998194814127"></a><a name="p7998194814127"></a><em id="i199987481127"><a name="i199987481127"></a><a name="i199987481127"></a>需要把modelArts/目录下的start.py启动脚本拷贝到根目录下。</em></p>
       </div></div>
       </td>
       </tr>
       <tr id="row59981448101210"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p19998124812123"><a name="p19998124812123"></a><a name="p19998124812123"></a><em id="i1399864831211"><a name="i1399864831211"></a><a name="i1399864831211"></a>输入数据配置</em></p>
       </td>
       <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p139982484129"><a name="p139982484129"></a><a name="p139982484129"></a><em id="i299816484122"><a name="i299816484122"></a><a name="i299816484122"></a>代码路径参数：data_dir</em></p>
       </td>
       </tr>
       <tr id="row179981948151214"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p89981948191220"><a name="p89981948191220"></a><a name="p89981948191220"></a><em id="i599844831217"><a name="i599844831217"></a><a name="i599844831217"></a>输出数据配置</em></p>
       </td>
       <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p599814485120"><a name="p599814485120"></a><a name="p599814485120"></a><em id="i189981748171218"><a name="i189981748171218"></a><a name="i189981748171218"></a>代码路径参数：train_url</em></p>
       </td>
       </tr>
       </tbody>
       </table>

### 创建训练作业

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“训练管理 \> 训练作业（New）”，默认进入“训练作业”列表。
2. 单击“创建训练作业”，进入“创建训练作业”页面，在该页面填写训练作业相关参数。

    1. 填写基本信息。

        基本信息包含“名称”和“描述”。

    2. 填写作业参数。

        包含数据来源、算法来源等关键信息。本步骤只提供训练任务部分参数配置说明，其他参数配置详情请参见[《ModelArts AI 工程师用户指南](https://support.huaweicloud.com/modelarts/index.html)》中“训练模型（new）”。

        ![输入图片说明](images/7.png)

        **表 5**  参数说明

        <a name="table96111035134613"></a>
        <table><thead align="left"><tr id="zh-cn_topic_0000001178072725_row1727593212228"><th class="cellrowborder" valign="top" width="20%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001178072725_p102751332172212"><a name="zh-cn_topic_0000001178072725_p102751332172212"></a><a name="zh-cn_topic_0000001178072725_p102751332172212"></a>参数名称</p>
        </th>
        <th class="cellrowborder" valign="top" width="20%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001178072725_p186943411156"><a name="zh-cn_topic_0000001178072725_p186943411156"></a><a name="zh-cn_topic_0000001178072725_p186943411156"></a>子参数</p>
        </th>
        <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001178072725_p1827543282216"><a name="zh-cn_topic_0000001178072725_p1827543282216"></a><a name="zh-cn_topic_0000001178072725_p1827543282216"></a>说明</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="zh-cn_topic_0000001178072725_row780219161358"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p0803121617510"><a name="zh-cn_topic_0000001178072725_p0803121617510"></a><a name="zh-cn_topic_0000001178072725_p0803121617510"></a>算法</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p186947411520"><a name="zh-cn_topic_0000001178072725_p186947411520"></a><a name="zh-cn_topic_0000001178072725_p186947411520"></a>我的算法</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p20803141614514"><a name="zh-cn_topic_0000001178072725_p20803141614514"></a><a name="zh-cn_topic_0000001178072725_p20803141614514"></a>选择“我的算法”页签，勾选上文中创建的算法。</p>
        <p id="zh-cn_topic_0000001178072725_p24290418284"><a name="zh-cn_topic_0000001178072725_p24290418284"></a><a name="zh-cn_topic_0000001178072725_p24290418284"></a>如果没有创建算法，请单击“创建”进入创建算法页面，详细操作指导参见“创建算法”。</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row1927503211228"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p327583216224"><a name="zh-cn_topic_0000001178072725_p327583216224"></a><a name="zh-cn_topic_0000001178072725_p327583216224"></a>训练输入</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1069419416510"><a name="zh-cn_topic_0000001178072725_p1069419416510"></a><a name="zh-cn_topic_0000001178072725_p1069419416510"></a>数据来源</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p142750323227"><a name="zh-cn_topic_0000001178072725_p142750323227"></a><a name="zh-cn_topic_0000001178072725_p142750323227"></a>选择OBS上数据集存放的目录。</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row127593211227"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p9744151562"><a name="zh-cn_topic_0000001178072725_p9744151562"></a><a name="zh-cn_topic_0000001178072725_p9744151562"></a>训练输出</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1027563212210"><a name="zh-cn_topic_0000001178072725_p1027563212210"></a><a name="zh-cn_topic_0000001178072725_p1027563212210"></a>模型输出</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p13275113252214"><a name="zh-cn_topic_0000001178072725_p13275113252214"></a><a name="zh-cn_topic_0000001178072725_p13275113252214"></a>选择训练结果的存储位置（OBS路径），请尽量选择空目录来作为训练输出路径。</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row18750142834916"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p5751172811492"><a name="zh-cn_topic_0000001178072725_p5751172811492"></a><a name="zh-cn_topic_0000001178072725_p5751172811492"></a>规格</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p107514288495"><a name="zh-cn_topic_0000001178072725_p107514288495"></a><a name="zh-cn_topic_0000001178072725_p107514288495"></a>-</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p3751142811495"><a name="zh-cn_topic_0000001178072725_p3751142811495"></a><a name="zh-cn_topic_0000001178072725_p3751142811495"></a>Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row16275103282219"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p15275132192213"><a name="zh-cn_topic_0000001178072725_p15275132192213"></a><a name="zh-cn_topic_0000001178072725_p15275132192213"></a>作业日志路径</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1369484117516"><a name="zh-cn_topic_0000001178072725_p1369484117516"></a><a name="zh-cn_topic_0000001178072725_p1369484117516"></a>-</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p227563218228"><a name="zh-cn_topic_0000001178072725_p227563218228"></a><a name="zh-cn_topic_0000001178072725_p227563218228"></a>设置训练日志存放的目录。请注意选择的OBS目录有读写权限。</p>
        </td>
        </tr>
        </tbody>
        </table>

3. 单击“提交”，完成训练作业的创建。

    训练作业一般需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟到几十分钟不等。

### 查看训练任务日志

1. 在ModelArts管理控制台，在左侧导航栏中选择“训练管理 \> 训练作业（New）”，默认进入“训练作业”列表。
2. 在训练作业列表中，您可以单击作业名称，查看该作业的详情。

    详情中包含作业的基本信息、训练参数、日志详情和资源占用情况。

    ![输入图片说明](https://images.gitee.com/uploads/images/2021/0926/182445_c83450d1_8725359.png "Logs.png")

