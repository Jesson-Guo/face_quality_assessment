#!/usr/bin/env bash
set -e

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

export MX_SDK_HOME=/home/data/cd_mindx/mxManufacture-2.0.2/
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

dataset_path=$1
pipeline_path=$2

#to set PYTHONPATH, import the StreamManagerApi.py
export PYTHONPATH=$PYTHONPATH:${MX_SDK_HOME}/python

if [ ! -d infer_result ] ;then
    mkdir infer_result;
fi

# compile plugin program
if [ -d ../util/plugins/build ] ;then
    rm -rf ../util/plugins/build;
fi
cmake -S ../util/plugins -B ../util/plugins/build
make -C ../util/plugins/build -j

if [ ! -d $MX_SDK_HOME/lib/plugins/ ] ;then
    echo "distance plugin directory not exists";
    exit 0;
fi
if [ -f ../util/plugins/build/libmxpi_transposeplugin.so ] ;then
    echo "copy file libmxpi_transposeplugin.so to plugins";
    cp ../util/plugins/build/libmxpi_transposeplugin.so $MX_SDK_HOME/lib/plugins/;
    echo "copy successfully";
fi

echo "start infer...";
python3.7 -u fqa_opencv.py \
--dataset=$dataset_path \
--pipeline=$pipeline_path \
--output=infer_result > infer.log 2>&1
exit 0


python3.7 -u fqa_opencv.py --dataset=$dataset_path --pipeline=$pipeline_path --output=infer_result > infer.log 2>&1