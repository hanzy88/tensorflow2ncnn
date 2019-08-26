############################
Thank you for the share of ncnn, refer to https://github.com/Tencent/ncnn
############################

This repository is mainly for the converting from tensorflow to ncnn directly based on the NCNN for Tensorflow https://github.com/jiangxiluning/ncnn-tensorflow.

First build the ncnn:

Since the related cmakefiles have changed to rebuilt on my machine, you can follow the steps in ncnn to build the ncnn.

And for the convert for tensorflow:

please check the related files in ./tools/tensorflow/, tensorflow2ncnn.cpp can convert the pb file to the ncnn.param and ncnn.bin

And for the new layers built for tensorflow:

you can check the related files in ./src/layer, the new layers are started with tf* (forgive for my poor coding in the layer, it will be updated in the future).

How to build new layer for ncnn, please check:
https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step
