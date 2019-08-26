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

#include "tfrange.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(TFRange)

TFRange::TFRange()
{
    one_blob_only = false;
    support_inplace = false;
}

int TFRange::load_param(const ParamDict& pd)
{
    start = pd.get(0, 0);
    limit = pd.get(1, 1);
    delta = pd.get(2, 1);
    //fprintf(stderr, "slices: %d %d %d \n", start, limit, delta);
    return 0;
}


int TFRange::forward(const std::vector<Mat>& /*bottom_blobs*/, std::vector<Mat>& top_blobs, const Option& opt) const
{

    Mat& top_blob = top_blobs[0];
    size_t elemsize = 4u;

    int new_w = limit - start;
    if(delta != 1){
        new_w = (limit - start) / delta + 1;
    }

    top_blob.create(new_w, elemsize, opt.blob_allocator);

    float* outptr = top_blob;
    for(int i = 0;i < limit;i += delta){
        *outptr++ = (float)i;
        //fprintf(stderr, "%f\n", *outptr);
        //*outptr++;
    }

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
