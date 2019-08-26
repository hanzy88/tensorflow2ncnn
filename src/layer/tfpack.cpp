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

#include "tfpack.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(TFPack)

TFPack::TFPack()
{
    one_blob_only = true;
    support_inplace = false;
}

int TFPack::load_param(const ParamDict& pd)
{
    slices = pd.get(0, 1);
    //fprintf(stderr, "slices: %d\n", slices);

    return 0;
}

int TFPack::load_model(const ModelBin& mb)
{
    pack = mb.load(slices, 1);
    if (pack.empty())
        return -100; 

    return 0;
}

int TFPack::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const Mat& pack_blob = pack;

    size_t elemsize = bottom_blob.elemsize + pack_blob.elemsize;

    int _w = bottom_blob.w;
    int _h = bottom_blob.h;
    int _c = bottom_blob.c;

    int new_h = _h + pack_blob.h;

    top_blob.create(_w, new_h, _c, elemsize, opt.blob_allocator);
    int q = 0;
    const int *tmp = (const int*)pack_blob;

    float* outptr = top_blob;
    for(size_t i = 0;i < 2;i++){
        if(*tmp <= 0){
            *outptr = -1;
            q++;
        }
        else{
            const float *ptr = (const float*)bottom_blob;
            fprintf(stderr, "pack2: %f\n", *ptr);
            memcpy(outptr + q, ptr, bottom_blob.elemsize);
            q += _w * bottom_blob.h * _c;
        }
        //fprintf(stderr, "pack1: %f\n", *ptr);
        
    }
    return 0;
}

} // namespace ncnn
