find_package(Protobuf REQUIRED)
protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS caffe.proto)
add_executable(caffe2ncnn caffe2ncnn.cpp ${CAFFE_PROTO_SRCS} ${CAFFE_PROTO_HDRS})
target_include_directories(caffe2ncnn
    PRIVATE
        ${Protobuf_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(caffe2ncnn PRIVATE cxx_std_11)
target_link_libraries(caffe2ncnn PRIVATE ${Protobuf_LIBRARIES})
