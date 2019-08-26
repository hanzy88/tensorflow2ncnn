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

#include "tfresizebilinear.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(TFResizeBilinear);

TFResizeBilinear::TFResizeBilinear()
{
    one_blob_only = true;
    support_inplace = false;
}

int TFResizeBilinear::load_param(const ParamDict& pd)
{
    output_height = pd.get(0, 0);
    output_width = pd.get(1, 0);

    return 0;
}

static void linear_coeffs(int w, int outw, int* xofs, float* alpha)
{
    double scale = (double)w / outw;

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        int sx = floor(fx);
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx*2    ] = 1.f - fx;
        alpha[dx*2 + 1] = fx;
    }
}

static void resize_bilinear_image(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++ )
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src.row(sy+1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src.row(sy);
            const float* S1 = src.row(sy+1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0]*a0 + S0p[1]*a1;
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);
        for (int dx = 0; dx < w; dx++)
        {
//             D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }
}



int TFResizeBilinear::forward(const Mat &bottom_blob, Mat &top_blob, const Option& opt) const
{
    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int c = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int oh = output_height;
    int ow = output_width;
    if (bottom_blob.dims == 1)
    {
        h = 1;
        w = 1;
        c = bottom_blob.w;
    }
    if (oh == h && ow == w)
    {
        top_blob = bottom_blob;
        return 0;
    }
    top_blob.create(ow, oh, c, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (bottom_blob.dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat top_blob_c = top_blob.channel(q);
            const float *ptr = ((const float*)bottom_blob.data + q);
            top_blob_c.fill(*ptr);
        }
        return 0;
    }

    int* buf = new int[ow + oh + ow*2 + oh*2];

    int* xofs = buf;//new int[ow];
    int* yofs = buf + ow;//new int[oh];

    float* alpha = (float*)(buf + ow + oh);//new float[ow * 2];
    float* beta = (float*)(buf + ow + oh + ow*2);//new float[oh * 2];

    linear_coeffs(w, ow, xofs, alpha);
    linear_coeffs(h, oh, yofs, beta);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; ++q)
    {
        const Mat src = bottom_blob.channel(q);
        Mat dst = top_blob.channel(q);

        resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
    }

    delete[] buf;

    return 0;
}

} // namespace ncnn
