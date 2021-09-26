#!/usr/bin/env bash
model_path=$1
output_model_name=$2

atc \
--input_format=NCHW \
--framework=1 \
--model=$model_path \
--output=$output_model_name \
--output_type=FP32 \
--soc_version=Ascend310