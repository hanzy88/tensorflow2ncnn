Update:
======
The yolov3 based on full cnn/ mobilenetv2 are checked successfully by tensorflow2ncnn. Since the model was only trained with 16 epoches AND accuracy loss after conversion, the result was not so good, as shown in follow:

![Image text](https://github.com/hanzy88/tensorflow2ncnn/blob/master/images/2eff677a1588b8bcf8382183192b44c.png)

Since the related flies are too big, you can download by baiduyun:

https://pan.baidu.com/s/1mQdPuTeiRw2iimcyxU0WYw 
code: 3hkh 

= =
If it's helpful for you, please give a star. 
And if you are interested in tf2ncnn, welcome to improve it together to make tf2ncnn better.

Tensorflow2ncnn
==============

###########
Thanks for the share of ncnn, refer to https://github.com/Tencent/ncnn
###########

This repository is mainly for the converting from tensorflow to ncnn directly based on the NCNN for Tensorflow https://github.com/jiangxiluning/ncnn-tensorflow.

First build the ncnn:

Since the related cmakefiles have changed to rebuilt on my machine, you can follow the steps in ncnn to build the ncnn.

And for the convert for tensorflow:

please check the related files in tools/tensorflow/, tensorflow2ncnn.cpp can convert the pb file to the ncnn.param and ncnn.bin

And for the new layers built for tensorflow:

you can check the related files in src/layer, the new layers are started with tf* (forgive for my poor coding in the layer, it will be updated in the future).

How to build new layer for ncnn, please check:

https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step

Once the new layer are added, add the layer to make in src/CMakefiles, like: ncnn_add_layer(TFReshape). and then rebuild the project.


For now, the problem caused by original bactchnorm in ncnn-tensorflow has been solved. 

And layer "Shape", "StridedSlice", "Pack", "ResizeBilinear", "LeakyRelu", "Relu6", "Range", "Tile", "Reshape", "Cast" has been added or updated for tensorflow. 

The normal CNN with FC has tested,  tf.flatten NOT support yet because the weight file cannot be aligned correctly, but you can use reshape op with the caculated shape like tf.reshape(max_pool, [-1, 64]).

