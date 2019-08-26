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

#include <float.h>
#include <stdio.h>
#include <limits.h>

#include <iostream>

#include <fstream>
#include <set>
#include <limits>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <google/protobuf/map.h>


#include "graph.pb.h"

#define Log_han fprintf(stderr, "file : %s function: %s line: %d\n", __FILE__, __FUNCTION__, __LINE__);

static bool is_const_request(const std::string& node_name){
    //{"Reshape/shape", "strided_slice/stack"}
    std::vector<std::string> request_node = {""};
    for(int i = 0;i < request_node.size();i++){
        if(node_name.find(request_node[i])!= std::string::npos)
            return true;
    }
    return false;
}

static bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

static bool find_tensor_proto(const std::map<std::string, tensorflow::TensorProto>& weights,
                              const tensorflow::NodeDef& node, tensorflow::TensorProto& tensor)
{
    for (int j=0; j<node.input_size(); j++)
    {
        const std::string& input_name = node.input(j);

        const std::map<std::string, tensorflow::TensorProto>::const_iterator it = weights.find(input_name);
        if (it != weights.end())
        {
            tensor = it->second;
            return true;
        }
    }

    return false;
}

static bool get_tensor_proto(const std::map<std::string, tensorflow::TensorProto>& consts,
                             const tensorflow::NodeDef& node, tensorflow::TensorProto& tensor)
{
    const std::string& output_name = node.name();

    const std::map<std::string, tensorflow::TensorProto>::const_iterator it = consts.find(output_name);
    if (it != consts.end())
    {
        tensor = it->second;
        return true;
    }

    return false;
}


// for different AttrValue.value_case(), the value can obtained by s(string), i(int)
// f(float), b(bool), type, shape, tensor, list(.s(.s_size>0, string), .i(int), .f(float))
static float get_node_attr_f(const tensorflow::NodeDef& node, const char* key, float def = 0.f)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    if (it != attr.end())
    {
        // it->second.value_case() == 4
        //fprintf(stderr, "%f\n", it->second.f());
        def = it->second.f();
        return def;
        
    }

    return def;
}

static float get_node_attr_i(const tensorflow::NodeDef& node, const char* key, int def = 0)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    //Log_han;
    if (it != attr.end())
    {
        //Log_han;
        // it->second.value_case() == 4
        def = it->second.i();
        return def;
        
    }

    return def;
}

static std::string get_node_attr_s(const tensorflow::NodeDef& node, const char* key, const std::string& def = std::string())
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);

    if (it != attr.end())
    {
        //Log_han;
        // it->second.value_case() == 4
        return it->second.s();
        
    }

    return def;
}

static tensorflow::TensorProto get_node_attr_tensor(const tensorflow::NodeDef& node, const char* key)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);

    if (it != attr.end())
    {
        //Log_han;
        // it->second.value_case() == 4
        return it->second.tensor();
        
    }

    return tensorflow::TensorProto();
}

static std::vector<int> get_node_attr_ai(const tensorflow::NodeDef& node, const char* key)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    
    std::vector<int> v;

    if (it != attr.end())
    {
        v.resize(it->second.list().i_size());
        for (int i=0; i<it->second.list().i_size(); i++)
        {
            v[i] = it->second.list().i(i);
        }
    }


    return v;
}

static std::vector<float> get_node_attr_af(const tensorflow::NodeDef& node, const char* key)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);

    std::vector<float> v;

    if (it != attr.end())
    {
        v.resize(it->second.list().f_size());
        for (int i=0; i<it->second.list().f_size(); i++)
        {
            v[i] = it->second.list().f(i);
        }
    }

    return v;
}



static bool find_attr_value(const tensorflow::NodeDef& node, const char* key, tensorflow::AttrValue& value)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    if (it != attr.end())
    {
        value = it->second;
        return true;
    }

    return false;
}

static int parse_tensor_reduction_dim(const tensorflow::TensorProto& tensor)
{
    int dim = 0;

    // dim == 0 // w h c -> X X X
    // dim == 1 // w h c -> X X c
    // dim == 2 // w h c -> X h c
    // dim == -1 // w h c -> w X X
    // dim == -2 // w h c -> w h X

    if (!tensor.tensor_content().empty() && tensor.dtype() == 3)// int32
    {
        const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
        int size = tensor.tensor_content().size() / sizeof(int);

        // n h w c
        // n h w
        // n w
        // TODO investigate two stage / three stage reduction
        if (size == 2)
        {
            if (data[0] == 1 && data[1] == 2)
            {
                dim = 1;
            }
        }
    }
    else
    {
        int axis = tensor.int_val(0);
        if (axis == 1)
            dim = 0;
        else if (axis == 3)
            dim = -2;
    }

    return dim;
}

int main(int argc, char** argv)
{
    const char* tensorflowpb = argv[1];
    const char* ncnn_prototxt = argc >= 4 ? argv[2] : "ncnn.param";
    const char* ncnn_modelbin = argc >= 4 ? argv[3] : "ncnn.bin";

    tensorflow::GraphDef graph;

    // load
    bool s1 = read_proto_from_binary(tensorflowpb, &graph);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    // magic
    fprintf(pp, "7767517\n");

    int node_count = graph.node_size();

//     fprintf(stderr, "node_count = %d\n\n", node_count);

    // node reference
    std::map<std::string, int> node_reference;

    // mapping for Const and Const-Identity
    std::map<std::string, tensorflow::TensorProto> weights;

    // Dropout like Identity
    std::set<std::string> dropouts;

    // Const before BinaryOp
    std::map<std::string, tensorflow::TensorProto> binaryop_consts;

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (int i=0; i<node_count; i++)
    {
        const tensorflow::NodeDef& node = graph.node(i);

        const std::string& output_name = node.name();

        if (node.op() == "Const")
        {
            tensorflow::AttrValue value;
            if (find_attr_value(node, "value", value))
            {
                //fprintf(stderr, "%s %d\n", node.name().c_str(), value.value_case());
                const tensorflow::TensorProto& tensor = value.tensor();
                weights[output_name] = tensor;
            }
            continue;
        }
        else if (node.op() == "Identity")
        {
            const std::string& input_name = node.input(0);
            if (weights.find(input_name) != weights.end())
            {
                weights[output_name] = weights[input_name];
                continue;
            }
            else
            {
                dropouts.insert(output_name);
            }
        }
        else if (node.op() == "NoOp")
        {
            weights[output_name] = tensorflow::TensorProto();
            continue;
        }
        else
        {
            bool isBinaryOp = false;
            if (node.op() == "Add" || node.op() == "BiasAdd" || node.op() == "Div"
                || node.op() == "Mul" || node.op() == "RealDiv" || node.op() == "Sub")
            {
                isBinaryOp = true;
            }
            if (node.op() == "Max" || node.op() == "Maximum" || node.op() == "Min" || node.op() == "Minimum")
            {
                // check weights
                tensorflow::TensorProto tensor;
                if (!find_tensor_proto(weights, node, tensor))
                {
                    isBinaryOp = true;
                }
            }

            if (isBinaryOp)
            {
                // check weights
                for (int j=0; j<node.input_size(); j++)
                {
                    const std::string& input_name = node.input(j);

                    std::map<std::string, tensorflow::TensorProto>::iterator it = weights.find(input_name);
                    if (it != weights.end())
                    {
                        // binary op with const, insert MemoryData layer and const blob
                        binaryop_consts[input_name] = it->second;
                        weights.erase(it);
                    }
                }
            }
        }

        // input
        for (int j=0; j<node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);
//             fprintf(stderr, "input = %s\n", input_name.c_str());

            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            blob_names.insert(input_name);

            if (node_reference.find(input_name) == node_reference.end())
            {
                node_reference[input_name] = 1;
            }
            else
            {
                node_reference[input_name] = node_reference[input_name] + 1;
            }
        }

        // output
//         fprintf(stderr, "output = %s\n", output_name.c_str());
        blob_names.insert(output_name);
    }

    // remove node_reference entry with reference equals to one
    int splitncnn_blob_count = 0;
    std::map<std::string, int>::iterator it = node_reference.begin();
    while (it != node_reference.end())
    {
        if (it->second == 1)
        {
            node_reference.erase(it++);
        }
        else
        {
            splitncnn_blob_count += it->second;
//             fprintf(stderr, "%s %d\n", it->first.c_str(), it->second);
            ++it;
        }
    }

    fprintf(pp, "%lu %lu\n", node_count + node_reference.size() - weights.size(), blob_names.size() + splitncnn_blob_count);

    int internal_split = 0;

    std::ofstream write;
    write.open("node_layer.txt");

    for (int i=0; i<node_count; i++)
    {
        const tensorflow::NodeDef& node = graph.node(i);

        
        write << "layer name: "<< node.name().c_str() << "    " << i << std::endl;

        //fprintf(stderr, "layer name: %s %d\n", node.name().c_str(), i);

        // layer definition line, repeated
        // [type] [name] [bottom blob count] [top blob count] [bottom blobs] [top blobs] [layer specific params]
//         fprintf(pp, "%-16s %-16s %d %d", layer.type().c_str(), layer.name().c_str(), node.input_size(), layer.top_size());

        if (node.op() == "Add" || node.op() == "BiasAdd")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (node.op() == "AvgPool")
        {
            fprintf(pp, "%-16s", "Pooling");
        }
        else if (node.op() == "Concat" || node.op() == "ConcatV2")
        {
            fprintf(pp, "%-16s", "Concat");
        }
        else if (node.op() == "Const")
        {
            // check before binaryop
            tensorflow::TensorProto tensor;
            if (get_tensor_proto(binaryop_consts, node, tensor))
            {
                fprintf(pp, "%-16s", "MemoryData");
            }
            else
            {
                continue;
            }
        }
        else if (node.op() == "Conv2D")
        {
            fprintf(pp, "%-16s", "Convolution");
        }
        else if (node.op() == "DepthwiseConv2dNative")
        {
            fprintf(pp, "%-16s", "ConvolutionDepthWise");
        }
        else if (node.op() == "Div" || node.op() == "RealDiv")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (node.op() == "Exp")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "ExpandDims")
        {
            fprintf(pp, "%-16s", "ExpandDims");
        }
        else if (node.op() == "Floor")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "Identity")
        {
            // check before binaryop
            tensorflow::TensorProto tensor;
            if (get_tensor_proto(binaryop_consts, node, tensor))
            {
                fprintf(pp, "%-16s", "MemoryData");
            }
            else if (dropouts.find(node.name()) != dropouts.end())
            {
                fprintf(pp, "%-16s", "Dropout");
            }
            else
            {
                continue;
            }
        }
        else if (node.op() == "LRN")
        {
            fprintf(pp, "%-16s", "LRN");
        }
        else if (node.op() == "MatMul")
        {
            fprintf(pp, "%-16s", "InnerProduct");
        }
        else if (node.op() == "Max" || node.op() == "Maximum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                fprintf(pp, "%-16s", "Reduction");
            }
            else
            {
                fprintf(pp, "%-16s", "BinaryOp");
            }
        }
        else if (node.op() == "MaxPool")
        {
            fprintf(pp, "%-16s", "Pooling");
        }
        else if (node.op() == "Min" || node.op() == "Minimum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                fprintf(pp, "%-16s", "Reduction");
            }
            else
            {
                fprintf(pp, "%-16s", "BinaryOp");
            }
        }
        else if (node.op() == "Mul")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (node.op() == "Neg")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "NoOp")
        {
            continue;
        }
        else if (node.op() == "Pad")
        {
            fprintf(pp, "%-16s", "Padding");
        }
        else if (node.op() == "Placeholder")
        {
            fprintf(pp, "%-16s", "Input");
        }
        else if (node.op() == "Prod")
        {
            fprintf(pp, "%-16s", "Reduction");
        }
        else if (node.op() == "Reciprocal")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "Relu")
        {
            fprintf(pp, "%-16s", "ReLU");
        }
        else if (node.op() == "Rsqrt")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "Sigmoid")
        {
            fprintf(pp, "%-16s", "Sigmoid");
        }
        else if (node.op() == "Softmax")
        {
            fprintf(pp, "%-16s", "Softmax");
        }
        else if (node.op() == "Square")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "Squeeze")
        {
            fprintf(pp, "%-16s", "Squeeze");
        }
        else if (node.op() == "Sub")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (node.op() == "Sum")
        {
            fprintf(pp, "%-16s", "Reduction");
        }
        else if (node.op() == "FusedBatchNorm")
        {
            fprintf(pp, "%-16s", "BatchNorm");
        }
        else if (node.op() == "Shape")
        {
            fprintf(pp, "%-16s", "TFShape");
        }
        else if (node.op() == "Reshape")
        {
            fprintf(pp, "%-16s", "TFReshape");
        }
        else if (node.op() == "StridedSlice"){
            //Log_han;
            fprintf(pp, "%-16s", "TFStridedSlice");
            //fprintf(pp, "%-16s", "StridedSlice");
        }
        else if (node.op() == "Pack"){
            //Log_han;
            fprintf(pp, "%-16s", "TFPack");
        }
        else if(node.op() == "ResizeBilinear"){
            fprintf(pp, "%-16s", "TFResizeBilinear");
        }
        else if(node.op() == "LeakyRelu"){
            fprintf(pp, "%-16s", "ReLU");
        }
        else if(node.op() == "Relu6"){
            fprintf(pp, "%-16s", "Clip");
        }
        else if(node.op() == "Range"){
            fprintf(pp, "%-16s", "TFRange");
        }
        else if(node.op() == "Tile"){
            fprintf(pp, "%-16s", "TFTile");
        }
        else if(node.op() == "Cast"){
            fprintf(pp, "%-16s", "Cast");
        }
        else
        {
            fprintf(pp, "%-16s", node.op().c_str());
            fprintf(stderr, "%s not supported yet !\nn", node.op().c_str());
        }

        int input_size = node.input_size();
        for (int j=0; j<node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);
            if (weights.find(input_name) != weights.end())
            {
                input_size--;
            }
        }

        fprintf(pp, " %-32s %d 1", node.name().c_str(), input_size);

        for (int j=0; j<node.input_size(); j++)
        {
            std::string input_name = node.input(j);

            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            if (node_reference.find(input_name) != node_reference.end())
            {
                int refidx = node_reference[input_name] - 1;
                node_reference[input_name] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                input_name = input_name + splitsuffix;
            }

            fprintf(pp, " %s", input_name.c_str());
        }

        fprintf(pp, " %s", node.name().c_str());

        if (node.op() == "Add" || node.op() == "BiasAdd")
        {
            int op_type = 0;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "AvgPool")
        {
            int pooling_type = 1;

            int kernel_size_h = 1;
            int kernel_size_w = 1;
            int stride_h = 1;
            int stride_w = 1;
            int pad = 0;

            int global_pooling = 0;
            int pad_mode = 1;

            tensorflow::AttrValue value_ksize;
            if (find_attr_value(node, "ksize", value_ksize))
            {
                // batch, height, width, channels
                kernel_size_h = value_ksize.list().i(1);
                kernel_size_w = value_ksize.list().i(2);
            }

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad_mode = 1;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad_mode = 2;
                }
            }

            fprintf(pp, " 0=%d", pooling_type);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 11=%d", kernel_size_h);
            fprintf(pp, " 2=%d", stride_w);
            fprintf(pp, " 12=%d", stride_h);
            fprintf(pp, " 3=%d", pad);
            fprintf(pp, " 4=%d", global_pooling);
            fprintf(pp, " 5=%d", pad_mode);
        }
        else if (node.op() == "Concat" || node.op() == "ConcatV2" || node.op() == "Shape")
        {
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                // TODO
//                 int axis = tensor.int_val(0);
            }
        }
        else if (node.op() == "Const" || node.op() == "Identity")
        {
            // check before binaryop
            tensorflow::TensorProto tensor;
            if (get_tensor_proto(binaryop_consts, node, tensor))
            {
                const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

                int w = 0;
                int h = 0;
                int c = 0;

                if (shape.dim_size() == 1)
                {
                    w = shape.dim(0).size();
                }
                else if (shape.dim_size() == 2)
                {
                    h = shape.dim(0).size();
                    w = shape.dim(1).size();
                }
                else if (shape.dim_size() == 3)
                {
                    c = shape.dim(2).size();
                    h = shape.dim(0).size();
                    w = shape.dim(1).size();
                }

                int weight_data_size = 0;

                if (!tensor.tensor_content().empty())
                {
                    if (tensor.dtype() == 1)// float
                    {
                        const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                        weight_data_size = tensor.tensor_content().size() / sizeof(float);

                        if (c == 0)
                            fwrite(data, sizeof(float), weight_data_size, bp);
                        else
                        {
                            float tmp;
                            // h-w-c to c-h-w
                            for (int p=0; p<c; p++)
                            {
                                for (int i=0; i<h; i++)
                                {
                                    for (int j=0; j<w; j++)
                                    {
                                        tmp = data[i*w*c + j*c + p];
                                        fwrite(&tmp, sizeof(float), 1, bp);
                                    }
                                }
                            }
                        }
                    }
                    else if (tensor.dtype() == 3)// int32
                    {
                        const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                        weight_data_size = tensor.tensor_content().size() / sizeof(int);

                        float tmp;
                        if (c == 0)
                        {
                            for (int i=0; i<weight_data_size; i++)
                            {
                                tmp = data[i];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                        else
                        {
                            // h-w-c to c-h-w
                            for (int p=0; p<c; p++)
                            {
                                for (int i=0; i<h; i++)
                                {
                                    for (int j=0; j<w; j++)
                                    {
                                        tmp = data[i*w*c + j*c + p];
                                        fwrite(&tmp, sizeof(float), 1, bp);
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    if (tensor.dtype() == 1)// float
                    {
                        float val = tensor.float_val(0);
                        fwrite(&val, sizeof(float), 1, bp);
                    }
                    else if (tensor.dtype() == 3)// int32
                    {
                        float val = tensor.int_val(0);
                        fwrite(&val, sizeof(float), 1, bp);
                    }
                }

                fprintf(pp, " 0=%d", w);
                fprintf(pp, " 1=%d", h);
                fprintf(pp, " 2=%d", c);
            }
        }
        else if (node.op() == "Conv2D")
        {
            // weights
            tensorflow::TensorProto tensor;
            find_tensor_proto(weights, node, tensor);

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int kernel_size_h = shape.dim(0).size();
            int kernel_size_w = shape.dim(1).size();
            int num_input = shape.dim(2).size();
            int num_output = shape.dim(3).size();

            int stride_h = 1;
            int stride_w = 1;
            int dilation_h = 1;
            int dilation_w = 1;
            int pad = 0;

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad = 0;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad = -233;
                }
            }

            tensorflow::AttrValue value_rate;
            if (find_attr_value(node, "rate", value_rate))
            {
                // height, width
                dilation_h = value_rate.list().i(0);
                dilation_w = value_rate.list().i(1);
            }

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder h-w-i-o to o-i-h-w
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*num_input*num_output + j*num_input*num_output + q*num_output + p];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
                else if (node.op() == "StridedSlice"){
                    //Log_han;
                    // To do;
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*num_input*num_output + j*num_input*num_output + q*num_output + p];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
            }

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 11=%d", kernel_size_h);
            fprintf(pp, " 2=%d", dilation_w);
            fprintf(pp, " 12=%d", dilation_h);
            fprintf(pp, " 3=%d", stride_w);
            fprintf(pp, " 13=%d", stride_h);
            fprintf(pp, " 4=%d", pad);
            fprintf(pp, " 5=%d", bias_term);
            fprintf(pp, " 6=%d", weight_data_size);
        }
        else if (node.op() == "DepthwiseConv2dNative")
        {
            // weights
            tensorflow::TensorProto tensor;
            find_tensor_proto(weights, node, tensor);

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int kernel_size_h = shape.dim(0).size();
            int kernel_size_w = shape.dim(1).size();
            int num_input = shape.dim(2).size();
            int channel_multiplier = shape.dim(3).size();

            int num_output = num_input * channel_multiplier;
            int group = num_input;

            int stride_h = 1;
            int stride_w = 1;
            int dilation_h = 1;
            int dilation_w = 1;
            int pad = 0;

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad = 0;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad = -233;
                }
            }

            tensorflow::AttrValue value_rate;
            if (find_attr_value(node, "rate", value_rate))
            {
                // height, width
                dilation_h = value_rate.list().i(0);
                dilation_w = value_rate.list().i(1);
            }

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder h-w-i-cm to i-cm-h-w
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_input; p++)
                    {
                        for (int q=0; q<channel_multiplier; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*channel_multiplier*num_input + j*channel_multiplier*num_input + p*channel_multiplier + q];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_input; p++)
                    {
                        for (int q=0; q<channel_multiplier; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*channel_multiplier*num_input + j*channel_multiplier*num_input + p*channel_multiplier + q];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
            }

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 11=%d", kernel_size_h);
            fprintf(pp, " 2=%d", dilation_w);
            fprintf(pp, " 12=%d", dilation_h);
            fprintf(pp, " 3=%d", stride_w);
            fprintf(pp, " 13=%d", stride_h);
            fprintf(pp, " 4=%d", pad);
            fprintf(pp, " 5=%d", bias_term);
            fprintf(pp, " 6=%d", weight_data_size);
            fprintf(pp, " 7=%d", group);
        }
        else if (node.op() == "Div" || node.op() == "RealDiv")
        {
            int op_type = 3;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Exp")
        {
            int op_type = 7;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "ExpandDims")
        {
            int expand_w = 0;
            int expand_h = 0;
            int expand_c = 0;

            tensorflow::AttrValue value_dim;
            if (find_attr_value(node, "Tdim", value_dim))
            {
                int dim = value_dim.i();
                if (dim == 0)
                    expand_w = 1;
                if (dim == 1)
                    expand_h = 1;
                if (dim == 2)
                    expand_c = 1;
            }

            fprintf(pp, " 0=%d", expand_w);
            fprintf(pp, " 1=%d", expand_h);
            fprintf(pp, " 2=%d", expand_c);
        }
        else if (node.op() == "Floor")
        {
            int op_type = 2;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "LRN")
        {
            int norm_region = 0;
            int local_size = 1;
            float alpha = 1.f;
            float beta = 0.5f;

            tensorflow::AttrValue value_depth_radius;
            if (find_attr_value(node, "depth_radius", value_depth_radius))
            {
                local_size = value_depth_radius.i() * 2 + 1;
            }

            tensorflow::AttrValue value_alpha;
            if (find_attr_value(node, "alpha", value_alpha))
            {
                alpha = value_alpha.f();
            }

            tensorflow::AttrValue value_beta;
            if (find_attr_value(node, "beta", value_beta))
            {
                beta = value_beta.f();
            }

            // TODO
            float bias = 1.f;
            tensorflow::AttrValue value_bias;
            if (find_attr_value(node, "bias", value_bias))
            {
                bias = value_bias.f();
            }

            fprintf(pp, " 0=%d", norm_region);
            fprintf(pp, " 1=%d", local_size);
            fprintf(pp, " 2=%f", alpha);
            fprintf(pp, " 3=%f", beta);
        }
        else if (node.op() == "MatMul")
        {
            // weights
            tensorflow::TensorProto tensor;
            find_tensor_proto(weights, node, tensor);

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int num_input = shape.dim(0).size();
            int num_output = shape.dim(1).size();

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder i-o to o-i
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            tmp = data[q*num_output + p];
                            fwrite(&tmp, sizeof(float), 1, bp);
                        }
                    }
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            tmp = data[q*num_output + p];
                            fwrite(&tmp, sizeof(float), 1, bp);
                        }
                    }
                }
            }

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", bias_term);
            fprintf(pp, " 2=%d", weight_data_size);
        }
        else if (node.op() == "Max" || node.op() == "Maximum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                int operation = 4;
                int dim = 0;
                float coeff = 1.f;

                dim = parse_tensor_reduction_dim(tensor);

                fprintf(pp, " 0=%d", operation);
                fprintf(pp, " 1=%d", dim);
                fprintf(pp, " 2=%f", coeff);
            }
            else
            {
                int op_type = 4;
                fprintf(pp, " 0=%d", op_type);
            }
        }
        else if (node.op() == "MaxPool")
        {
            int pooling_type = 0;

            int kernel_size_h = 1;
            int kernel_size_w = 1;
            int stride_h = 1;
            int stride_w = 1;
            int pad = 0;

            int global_pooling = 0;
            int pad_mode = 1;

            tensorflow::AttrValue value_ksize;
            if (find_attr_value(node, "ksize", value_ksize))
            {
                // batch, height, width, channels
                kernel_size_h = value_ksize.list().i(1);
                kernel_size_w = value_ksize.list().i(2);
            }

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad_mode = 1;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad_mode = 2;
                }
            }

            fprintf(pp, " 0=%d", pooling_type);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 11=%d", kernel_size_h);
            fprintf(pp, " 2=%d", stride_w);
            fprintf(pp, " 12=%d", stride_h);
            fprintf(pp, " 3=%d", pad);
            fprintf(pp, " 4=%d", global_pooling);
            fprintf(pp, " 5=%d", pad_mode);
        }
        else if (node.op() == "Min" || node.op() == "Minimum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                int operation = 5;
                int dim = 0;
                float coeff = 1.f;

                dim = parse_tensor_reduction_dim(tensor);

                fprintf(pp, " 0=%d", operation);
                fprintf(pp, " 1=%d", dim);
                fprintf(pp, " 2=%f", coeff);
            }
            else
            {
                int op_type = 5;
                fprintf(pp, " 0=%d", op_type);
            }
        }
        else if (node.op() == "Mul")
        {
            int op_type = 2;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Neg")
        {
            int op_type = 1;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "NoOp")
        {
        }
        else if (node.op() == "Pad")
        {
            int top = 0;
            int bottom = 0;
            int left = 0;
            int right = 0;
            int type = 0;
            float value = 0.f;

            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                if (!tensor.tensor_content().empty() && tensor.dtype() == 3)// int32
                {
                    const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    int size = tensor.tensor_content().size() / sizeof(int);

                    if (size == 8)
                    {
                        // n h w c
                        top = data[2];
                        bottom = data[3];
                        left = data[4];
                        right = data[5];
                    }
                }
            }

            tensorflow::AttrValue value_Tpaddings;
            if (find_attr_value(node, "Tpaddings", value_Tpaddings))
            {
                type = value_Tpaddings.i();
            }

            tensorflow::AttrValue value_T;
            if (find_attr_value(node, "T", value_T))
            {
                value = value_T.f();
            }

            fprintf(pp, " 0=%d", top);
            fprintf(pp, " 1=%d", bottom);
            fprintf(pp, " 2=%d", left);
            fprintf(pp, " 3=%d", right);
            fprintf(pp, " 4=%d", type);
            fprintf(pp, " 5=%f", value);
        }
        else if (node.op() == "Placeholder")
        {
            // TODO pass through
            fprintf(pp, " 0=0 1=0 2=0");
        }
        else if (node.op() == "Prod")
        {
            int operation = 6;
            int dim = 0;
            float coeff = 1.f;

            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                dim = parse_tensor_reduction_dim(tensor);
            }

            fprintf(pp, " 0=%d", operation);
            fprintf(pp, " 1=%d", dim);
            fprintf(pp, " 2=%f", coeff);
        }
        else if (node.op() == "Reciprocal")
        {
            int op_type = 15;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Relu")
        {
            float slope = 0.f;
            fprintf(pp, " 0=%f", slope);
        }
        else if (node.op() == "Reshape")
        {
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                if (!tensor.tensor_content().empty() && tensor.dtype() == 3)// int32
                {
                    //fprintf(stderr, "%s\n", node.name().c_str());
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    int size = tensor.tensor_content().size() / sizeof(int);
                    // n h w c
                    // n h w
                    // n w
                    if(size == 5){
                        fprintf(pp, " 0=%d 1=%d 2=%d 3=%d", data[2], data[1], data[3], data[4]);
                    }
                    if (size == 4)
                    {
                        fprintf(pp, " 0=%d 1=%d 2=%d 3=0", data[2], data[1], data[3]);
                    }
                    if (size == 3)
                    {
                        fprintf(pp, " 0=%d 1=%d 2=-233 3=1", data[2], data[1]);
                    }
                    if (size == 2)
                    {
                        fprintf(pp, " 0=%d 1=-233 2=-233 3=1", data[1]);
                    }
                }
            }
            else
            {
                // pass through
                fprintf(pp, " 0=0 1=0 2=0 3=0");
            }
        }
        else if (node.op() == "Rsqrt")
        {
            int op_type = 6;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Sigmoid")
        {
        }
        else if (node.op() == "Softmax")
        {
        }
        else if (node.op() == "Square")
        {
            int op_type = 4;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Squeeze")
        {
            int squeeze_w = 0;
            int squeeze_h = 0;
            int squeeze_c = 0;

            tensorflow::AttrValue value_squeeze_dims;
            if (find_attr_value(node, "squeeze_dims", value_squeeze_dims))
            {
                for (int i = 0; i<value_squeeze_dims.list().i_size(); i++)
                {
                    int dim = value_squeeze_dims.list().i(i);
                    if (dim == 0)
                        squeeze_w = 1;
                    if (dim == 1)
                        squeeze_h = 1;
                    if (dim == 2)
                        squeeze_c = 1;
                }
            }

            fprintf(pp, " 0=%d", squeeze_w);
            fprintf(pp, " 1=%d", squeeze_h);
            fprintf(pp, " 2=%d", squeeze_c);
        }
        else if (node.op() == "Sub")
        {
            int op_type = 1;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Sum")
        {
            int operation = 0;
            int dim = 0;
            float coeff = 1.f;

            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                dim = parse_tensor_reduction_dim(tensor);
            }

            fprintf(pp, " 0=%d", operation);
            fprintf(pp, " 1=%d", dim);
            fprintf(pp, " 2=%f", coeff);
        }
        else if (node.op() == "FusedBatchNorm"){

            const tensorflow::TensorProto& scale = weights[node.input(1)];
            const tensorflow::TensorProto& B = weights[node.input(2)];
            const tensorflow::TensorProto& mean = weights[node.input(3)];
            const tensorflow::TensorProto& var = weights[node.input(4)];

            int channels = scale.tensor_shape().dim(0).size(); // data size
            //fprintf(stderr, "channels: %d\n", channels);
            int dtype = scale.dtype();

            switch (dtype){
                case 1: //float
                {
                    float * scale_tensor = (float *)malloc(sizeof(float) * channels);
                    float * mean_tensor = (float *)malloc(sizeof(float) * channels);
                    float * var_tensor = (float *)malloc(sizeof(float) * channels);
                    float * b_tensor = (float *)malloc(sizeof(float) * channels);
                    const float * scale_data = reinterpret_cast<const float *>(scale.tensor_content().c_str());
                    const float * mean_data = reinterpret_cast<const float *>(mean.tensor_content().c_str());
                    const float * var_data = reinterpret_cast<const float *>(var.tensor_content().c_str());
                    const float * b_data = reinterpret_cast<const float *>(B.tensor_content().c_str());
                    
                    for(int i=0;i<channels;i++){
                        scale_tensor[i] = *scale_data++;
                        mean_tensor[i] = *mean_data++;
                        var_tensor[i] = *var_data++;
                        b_tensor[i] = *b_data++;
                        //fprintf(stderr, "scale_data: %f\n", * scale_data);
                    }

                    fwrite(scale_tensor, sizeof(float), channels, bp);
                    fwrite(mean_tensor, sizeof(float), channels, bp);
                    fwrite(var_tensor, sizeof(float), channels, bp);
                    fwrite(b_tensor, sizeof(float), channels, bp);
                    break;
                }

                case 2: // double
                {
                    double * scale_tensor = (double *)malloc(sizeof(double) * channels);
                    const double * scale_data = reinterpret_cast<const double *>(scale.tensor_content().c_str());

                    for(int i=0;i<channels;i++){
                        scale_tensor[i] = *scale_data++;
                        //scale_tensor[i] = *scale.float_val().data();
                    }

                    fwrite(scale_tensor, sizeof(double), channels, bp);

                    const double * mean_data = reinterpret_cast<const double *>(mean.tensor_content().c_str());
                    const double * var_data = reinterpret_cast<const double *>(var.tensor_content().c_str());
                    const double * b_data = reinterpret_cast<const double *>(B.tensor_content().c_str());

                    fwrite(mean_data, sizeof(double), channels, bp);
                    fwrite(var_data, sizeof(double), channels, bp);
                    fwrite(b_data, sizeof(double), channels, bp);
                    break;
                }

                case 6: //half
                {
                    channels = (int) (scale.tensor_content().size() / 16);

                    const char * scale_data = reinterpret_cast<const char *>(scale.tensor_content().c_str());
                    const char * mean_data = reinterpret_cast<const char *>(mean.tensor_content().c_str());
                    const char * var_data = reinterpret_cast<const char *>(var.tensor_content().c_str());
                    const char * b_data = reinterpret_cast<const char *>(B.tensor_content().c_str());

                    fwrite(scale_data, 16, channels, bp);
                    fwrite(mean_data, 16, channels, bp);
                    fwrite(var_data, 16, channels, bp);
                    fwrite(b_data, 16, channels, bp);
                    break;
                }
                default:
                    std::cerr << "Type is not supported." << std::endl;

            }
            fprintf(pp, " 0=%d", channels);
            //fprintf(stderr, "%d\n", channels);
            tensorflow::AttrValue value_epsilon;
            if (find_attr_value(node, "epsilon", value_epsilon)){
                float epsilon = value_epsilon.f();
                fprintf(pp, " 1=%f", epsilon);
            }
        }
        else if (node.op() == "StridedSlice"){
            const tensorflow::TensorProto& begin = weights[node.input(1)];
            const tensorflow::TensorProto& end = weights[node.input(2)];
            const tensorflow::TensorProto& step = weights[node.input(3)];

            const int * begin_data = reinterpret_cast<const int *>(begin.int_val().begin());
            const int * end_data = reinterpret_cast<const int *>(end.int_val().begin());
            const int * step_data = reinterpret_cast<const int *>(step.int_val().begin()); 

            int size = begin.int_val().size();

            int * begin_tensor = (int *)malloc(sizeof(int) * size);
            int * end_tensor = (int *)malloc(sizeof(int) * size);
            int * step_tensor = (int *)malloc(sizeof(int) * size);

            for(int i = 0;i < size;i++){
                begin_tensor[i] = begin_data[i];
                end_tensor[i] = end_data[i];
                step_tensor[i] = step_data[i];
            }
            
            //const int * scale_data = reinterpret_cast<const int *>(B.int_val());

            //fprintf(stderr, "%d %d\n", *scale_data, size);

            fwrite(begin_tensor, sizeof(int), size, bp);
            fwrite(end_tensor, sizeof(int), size, bp);
            fwrite(step_tensor, sizeof(int), size, bp);

            fprintf(pp, " 0=%d", size);
    
        }
        else if(node.op() == "Relu6"){
            float min = 0.f;
            float max = 6.f;
            fprintf(pp, " 0=%f", min);
            fprintf(pp, " 1=%f", max);
        }
        else if(node.op() == "Pack"){
            const tensorflow::TensorProto& pack = weights[node.input(0)];

            const int * pack_data = reinterpret_cast<const int *>(pack.int_val().begin()); 

            //fprintf(stderr, "%d\n", *pack_data);

            int size = pack.int_val().size();

            int * pack_tensor = (int *)malloc(sizeof(int) * size);

            for(int i = 0;i < size;i++){
                pack_tensor[i] = pack_data[i];
            }


            fwrite(pack_tensor, sizeof(int), size, bp);

            fprintf(pp, " 0=%d", size);
        }
        else if(node.op() == "LeakyRelu"){
            float alpha = get_node_attr_f(node, "alpha", 0.01f);
            //fprintf(stderr, "alpha %f\n", alpha);
            fprintf(pp, " 0=%f", alpha);
        }
        else if (node.op() == "ResizeBilinear")
        {
            std::vector<int> scales;
            const tensorflow::TensorProto& scales_tp = weights[node.input(1)];
            //fprintf(stderr, "%s\n", node.input(1).c_str());
            const int* shape_data = reinterpret_cast<const int *>(scales_tp.tensor_content().c_str());
            //fprintf(stderr, "%s\n", scales_tp.tensor_content());
            int float_data_size = scales_tp.float_val().size();
            //float data is None, use raw data instead
            if (float_data_size == 0) {
                float_data_size = scales_tp.tensor_shape().dim(0).size(); 
                //fprintf(stderr, "%d\n", float_data_size);
            }

            for (int j=0; j<float_data_size; j++)
            {
                //fprintf(stderr, "%d\n", *shape_data++);
                scales.push_back(*shape_data++);
            }

            int output_height = scales[0];
            int output_width = scales[1];
            fprintf(pp, " 0=%d", output_height);
            fprintf(pp, " 1=%d", output_width);
        }
        else if(node.op() == "Range"){
            const tensorflow::TensorProto& start = weights[node.input(0)];
            const tensorflow::TensorProto& limit = weights[node.input(1)];
            const tensorflow::TensorProto& delta = weights[node.input(2)];

            const int * start_data = reinterpret_cast<const int *>(start.int_val().begin()); 
            const int * limit_data = reinterpret_cast<const int *>(limit.int_val().begin()); 
            const int * delta_data = reinterpret_cast<const int *>(delta.int_val().begin()); 


            fprintf(pp, " 0=%d", *start_data);
            fprintf(pp, " 1=%d", *limit_data);
            fprintf(pp, " 2=%d", *delta_data);
        }
        else if(node.op() == "Tile"){
            const tensorflow::TensorProto& dims = weights[node.input(1)];
            const int * dims_n = reinterpret_cast<const int *>(dims.tensor_content().c_str());

            int int_data_size = dims.int_val().size();

            int tile_dims = 0;
            int tile_times = 0;
            for(int i = 0;i < int_data_size;i++){
                if(*dims_n != 1){
                    tile_dims = i;
                    tile_times = *dims_n;
                }
                *dims_n++;
            }
            fprintf(pp, " 0=%d", tile_dims);
            fprintf(pp, " 1=%d", tile_times);
        }
        else if(node.op() == "Cast"){
            const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

            const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it_To = attr.find("DstT");
            const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it_From = attr.find("Truncate");
            //fprintf(stderr, "%d %d\n", it_To->second.type(), it_From->second.type());
            fprintf(pp, " 0=%d", it_From->second.type());
            fprintf(pp, " 1=%d", it_To->second.type());
        }
        else
        {
            const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

            google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.begin();
            for (; it != attr.end(); it++)
            {
                std::cerr << it->first << " #" << it->second.type() << std::endl;
            }
        }

        fprintf(pp, "\n");

        std::string output_name = node.name();
        if (node_reference.find(output_name) != node_reference.end())
        {
            int refcount = node_reference[output_name];
            if (refcount > 1)
            {
                char splitname[256];
                sprintf(splitname, "splitncnn_%d", internal_split);
                fprintf(pp, "%-16s %-32s %d %d", "Split", splitname, 1, refcount);
                fprintf(pp, " %s", output_name.c_str());

                for (int j=0; j<refcount; j++)
                {
                    fprintf(pp, " %s_splitncnn_%d", output_name.c_str(), j);
                }
                fprintf(pp, "\n");

                internal_split++;
            }
        }
    }

    fclose(pp);
    fclose(bp);

    write.close();

    fprintf(stderr, "The work tf2ncnn finished.\n");

    return 0;
}