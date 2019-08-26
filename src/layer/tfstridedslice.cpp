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

#include "tfstridedslice.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(TFStridedSlice)

TFStridedSlice::TFStridedSlice()
{
    one_blob_only = true;
    support_inplace = false;
}

int TFStridedSlice::load_param(const ParamDict& pd)
{
    slices = pd.get(0, 1);
    //fprintf(stderr, "slices: %d\n", slices);

    return 0;
}

int TFStridedSlice::load_model(const ModelBin& mb)
{
    slice_begin = mb.load(slices, 1);
    if (slice_begin.empty())
        return -100;

    slice_end = mb.load(slices, 1);
    if (slice_end.empty())
        return -100;

    slice_step = mb.load(slices, 1);
    if (slice_step.empty())
        return -100;    

    return 0;
}


int TFStridedSlice::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{

    const int* slices_ptr1 = (const int*)slice_begin;
    const int* slices_ptr2 = (const int*)slice_end;
    const int* step_ptr = (const int*)slice_step;
    //fprintf(stderr, "%d %d %d\n", *slices_ptr1, *slices_ptr2, *step_ptr);

    const int num = 1;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    //fprintf(stderr, "%d %d %d\n", *slices_ptr1, *slices_ptr2, *step_ptr);

    if(*slices_ptr1 == 0){
        if(*slices_ptr2 == 1){
            float total = 1;
            for(int i = 0;i < bottom_blob.h;i++){
                total *= bottom_blob[i];
                //fprintf(stdout, "bottom_blob: %f\n",bottom_blob[i]);
            }
            top_blob.create(num, elemsize, opt.blob_allocator);
            float* ptr = top_blob;
            * ptr = total;
            //fprintf(stderr, "stridedslice: %f %f\n",*ptr, total);
        }else{
            *slices_ptr1++;
            *slices_ptr2++;
        }
    }else if(*slices_ptr1 < 0){
        float total;
        if(dims == 1){
            total = bottom_blob[bottom_blob.w-1];
        }else if(dims == 2){
            total = bottom_blob[bottom_blob.w*bottom_blob.h - 1];
        }else if(dims == 3){
            total = bottom_blob[bottom_blob.w*bottom_blob.h*bottom_blob.c - 1];
        }
        top_blob.create(num, elemsize, opt.blob_allocator);
        float* ptr = top_blob;
        * ptr = total;
        //fprintf(stdout, "stridedslice: %f\n",*ptr);
    }

    return 0;
}

} // namespace ncnn
