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

#include "tfreshape.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(TFReshape)

TFReshape::TFReshape()
{
    one_blob_only = false;
    support_inplace = false;
}

int TFReshape::load_param(const ParamDict& pd)
{
    w = pd.get(0, -233);
    h = pd.get(1, -233);
    c = pd.get(2, -233);
    permute = pd.get(3, 0);

    ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;

    return 0;
}

int TFReshape::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    //fprintf(stderr, "bottom_blobs size: %d, h:%d\n", bottom_blobs.size(), bottom_blobs[1].h);
    Mat& top_blob = top_blobs[0];
    size_t elemsize = bottom_blob.elemsize;
    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c;
    //fprintf(stderr, "%d\n", permute);
    
    if(bottom_blobs.size() == 1){

        if (ndim == 1)
        {
            int _w = w;

            if (_w == 0)
                _w = bottom_blob.w;

            if (_w == -1)
                _w = total;

            if (permute == 1)
            {
                top_blob.create(_w, elemsize, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                // c-h-w to h-w-c
                float* ptr = top_blob;
                for (int i=0; i<bottom_blob.h; i++)
                {
                    for (int j=0; j<bottom_blob.w; j++)
                    {
                        for (int p=0; p<bottom_blob.c; p++)
                        {
                            const float* bptr = bottom_blob.channel(p);
                            *ptr++ = bptr[i*bottom_blob.w + j];
                        }
                    }
                }
            }
            else
            {
                top_blob = bottom_blob.reshape(_w, opt.blob_allocator);
            }
        }
        else if (ndim == 2)
        {
            int _w = w;
            int _h = h;

            if (_w == 0)
                _w = bottom_blob.w;
            if (_h == 0)
                _h = bottom_blob.h;

            if (_w == -1)
                _w = total / _h;
            if (_h == -1)
                _h = total / _w;

            top_blob = bottom_blob.reshape(_w, _h, opt.blob_allocator);
        }
        else if (ndim == 3)
        {
            int _w = w;
            int _h = h;
            int _c = c;

            if (_w == 0)
                _w = bottom_blob.w;
            if (_h == 0)
                _h = bottom_blob.h;
            if (_c == 0)
                _c = bottom_blob.c;

            if (_w == -1)
                _w = total / _c / _h;
            if (_h == -1)
                _h = total / _c / _w;
            if (_c == -1)
                _c = total / _h / _w;
            top_blob = bottom_blob.reshape(_w, _h, _c, opt.blob_allocator);
        }
    }else if(bottom_blobs.size() == 2){
        const float* ptr = (const float*)bottom_blobs[1];
        if(bottom_blobs[1].h == 1){

            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            size_t elemsize = bottom_blob.elemsize;
            int size = w * h;
            top_blob.create(size * channels, elemsize, opt.blob_allocator);

            int cnt = 0;
            Mat tmp = bottom_blob.reshape(size*channels, opt.blob_allocator);
            float* outptr = (float*)top_blob;
            if (top_blob.empty())
                return -100;

            //#pragma omp parallel for num_threads(opt.num_threads)
            for(int i = 0;i < size; i++){
                for(int q = 0;q < channels;q++){
                    
                    outptr[cnt++] = tmp[q * size + i];

                }
            }
        }else if (bottom_blobs[1].h == 2)
        {
            int _w = *ptr++;
            int _h = *ptr;
            //fprintf(stderr, "%d %d\n", _w, _h);
            // especially for the flatten op.
            if (_w == 0 || _h == 0){
                
                int w = bottom_blob.w;
                int h = bottom_blob.h;
                int channels = bottom_blob.c;
                size_t elemsize = bottom_blob.elemsize;
                int size = w * h;
                top_blob.create(size * channels, elemsize, opt.blob_allocator);

                int cnt = 0;
                Mat tmp = bottom_blob.reshape(size*channels, opt.blob_allocator);
                float* outptr = (float*)top_blob;
                if (top_blob.empty())
                    return -100;

                //#pragma omp parallel for num_threads(opt.num_threads)
                for(int i = 0;i < size; i++){
                    for(int q = 0;q < channels;q++){
                        
                        outptr[cnt++] = tmp[q * size + i];

                    }
                }
            }else{
                if (_w == -1)
                    _w = total / _h;
                if (_h == -1)
                    _h = total / _w;

                //fprintf(stderr, "%d %d\n", _w, _h);
                //top_blob.create(_w, _h, elemsize, opt.blob_allocator);

                top_blob = bottom_blob.reshape(_w, _h, opt.blob_allocator);
            }
        }
        else if (bottom_blobs[1].h == 3)
        {
            int _w = *ptr++;
            int _h = *ptr++;
            int _c = *ptr;

            if (_w == -1)
                _w = total / _c / _h;
            if (_h == -1)
                _h = total / _c / _w;
            if (_c == -1)
                _c = total / _h / _w;

            top_blob.create(_w, _h, _c, elemsize, opt.blob_allocator);
            top_blob = bottom_blob.reshape(_w, _h, _c, opt.blob_allocator);
        }
    }else{
        fprintf(stderr, "Not defined\n");
        return -100;
    }

    if (top_blobs.empty())
        return -100;

    return 0;
}

} // namespace ncnn
