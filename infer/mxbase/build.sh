#!/usr/bin/env bash
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${LD_LIBRARY_PATH}

dataset_path=$1
model_path=$2

if [ -d build ] ;then
    rm -rf build;
fi

cmake -S . -B build
make -C ./build -j

./build/fqa_opencv $dataset_path $model_path