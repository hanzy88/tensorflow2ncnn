// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <fstream>
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yoloDection.h"
#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};


static int detect_yolov3(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov3;

#if NCNN_VULKAN
    yolov3.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    // original pretrained model from https://github.com/eric612/MobileNet-YOLO
    // https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov3/mobilenet_yolov3_lite_deploy.prototxt
    // https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov3/mobilenet_yolov3_lite_deploy.caffemodel
    yolov3.load_param("ncnn.param");
    yolov3.load_model("ncnn.bin");

    const int target_size = 416;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    //const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    //const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    //in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov3.create_extractor();
    ex.set_num_threads(4);
    int num_class = 80;
    int num_box = 3;
    float confidence_threshold=0.01;
    float nms_threshold = 0.45;
    int mask_group_num = 3;

    float bias[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};
    float msk[9] = {6,7,8,3,4,5,0,1,2};
    float anchors[3] = {32, 16,16};

    ncnn::Mat biases(18, sizeof(bias), bias);
    ncnn::Mat mask(9, sizeof(msk), msk);
    ncnn::Mat anchors_scale(3, sizeof(anchors), anchors);
     
    yoloDection yolo(num_class,num_box,confidence_threshold,nms_threshold,biases, 
                        mask, anchors_scale,mask_group_num);

    ex.input("input_x", in);

    ncnn::Mat out1;
    ex.extract("yolo_head1/cnn/add", out1);
    ncnn::Mat out2;
    ex.extract("yolo_head2/cnn/add", out2);
    ncnn::Mat out3;
    ex.extract("yolo_head3/cnn/add", out3);
    /*
    fprintf(stderr, "width: %d\n", out1.w);
    fprintf(stderr, "height: %d\n", out1.h);
    fprintf(stderr, "depth: %d\n", out1.c);
    fprintf(stderr, "width: %d\n", out2.w);
    fprintf(stderr, "height: %d\n", out2.h);
    fprintf(stderr, "depth: %d\n", out2.c);
    fprintf(stderr, "width: %d\n", out3.w);
    fprintf(stderr, "height: %d\n", out3.h);
    fprintf(stderr, "depth: %d\n", out3.c);
    */
    std::vector<ncnn::Mat> all_out;
    all_out.push_back(out1);
    all_out.push_back(out2);
    all_out.push_back(out3);
    ncnn::Mat out;
    yolo.detection(all_out, out);
    //out = out.reshape(out.w, out.h, 3, -1);
    fprintf(stderr, "width: %d\n", out.w);
    fprintf(stderr, "height: %d\n", out.h);
    fprintf(stderr, "depth: %d\n", out.c);

    /*
    fprintf(stderr, "%f\n", out);
    
    std::ofstream write;
    write.open("text.txt");
    
    for(int i = 0;i < out.c * out.h * out.c;i++){
        //fprintf(stderr, "out: %f\n",  out[i]);
        write << out[i] << std::endl;
    }
    write.close();
    */
//     printf("%d %d %d\n", out.w, out.h, out.c);

    
    objects.clear();
    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }
    
    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    std::vector<Object> objects;
    detect_yolov3(m, objects);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    draw_objects(m, objects);

    return 0;
}
