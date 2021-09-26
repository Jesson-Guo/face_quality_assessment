export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/data/cd_mindx/mxManufacture-2.0.2/opensource/lib:${LD_LIBRARY_PATH}

cmake -S . -Bbuild
make -C ./build -j

./build/fqa_opencv /home/cd_mindx/FaceQualityAssessment/dataset/AFLW2000