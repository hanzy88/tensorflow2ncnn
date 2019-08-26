###########
Thanks for the share of ncnn, refer to https://github.com/Tencent/ncnn
###########

This repository is mainly for the converting from tensorflow to ncnn directly based on the NCNN for Tensorflow https://github.com/jiangxiluning/ncnn-tensorflow.

First build the ncnn:

Since the related cmakefiles have changed to rebuilt on my machine, you can follow the steps in ncnn to build the ncnn.

And for the convert for tensorflow:

please check the related files in ./tools/tensorflow/, tensorflow2ncnn.cpp can convert the pb file to the ncnn.param and ncnn.bin

And for the new layers built for tensorflow:

you can check the related files in ./src/layer, the new layers are started with tf* (forgive for my poor coding in the layer, it will be updated in the future).

How to build new layer for ncnn, please check:
https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step


For now, the problem caused by original bactchnorm in ncnn-tensorflow has been solved. 

And layer "Shape", "StridedSlice", "Pack", "ResizeBilinear", "LeakyRelu", "Relu6", "Range", "Tile", "Reshape", "Cast" has been added or updated for tensorflow. 

I have test the normal CNN with FC,  tf.flatten NOT support yet because the weight file cannot be aligned correctly, but you can use reshape op with the caculated shape like tf.reshape(max_pool, [-1, 64]).

And the yolov3 based on mobilenetv2 also can output successfully based on the related files in projects. Once the results of output are checked correctly, the details will be updated.
