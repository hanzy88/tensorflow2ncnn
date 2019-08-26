// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tfshape.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(TFShape)

TFShape::TFShape()
{
    one_blob_only = true;
    support_inplace = false;
}


int TFShape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    size_t elemsize = 4u;
    const int num = 1;
    int ndim = 3;
    if(bottom_blob.c == 1){
        ndim = 2;
        if(bottom_blob.h == 1){
            ndim = 1;
        }
    }

    top_blob.create(num, ndim, elemsize, opt.blob_allocator);

    if (ndim == 1){
        float _w = bottom_blob.w;


        if (top_blob.empty())
            return -100;
        float* ptr = top_blob;
        * ptr = _w;      

    }else if (ndim == 2){
        float _w = bottom_blob.w;

        float _h = bottom_blob.h;

        if (top_blob.empty())
            return -100;
        float* ptr = top_blob;
        * ptr++ = _w; 
        * ptr = _h;  

    }else if (ndim == 3){

        float _w = bottom_blob.w;

        float _h = bottom_blob.h;

        float _c = bottom_blob.c;

        if (top_blob.empty())
            return -100;

        float* ptr = top_blob;
        * ptr++ = _w; 
        * ptr++ = _h;
        * ptr = _c; 
    }
    if (top_blob.empty())
            return -100;
    return 0;
}

} // namespace ncnn
