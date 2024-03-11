#!/bin/bash

export ENABLE_ORT_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON
export ENABLE_TEXT=ON
export ENABLE_TRT_BACKEND=ON
export WITH_GPU=ON
export TRT_DIRECTORY=/mnt/data/linsong/github/3rdparty/TensorRT-8.6.1.6
export CUDA_DIRECTORY=/usr/local/cuda
export PADDLEINFERENCE_DIRECTORY=/mnt/data/linsong/github/3rdparty/paddle_inference
