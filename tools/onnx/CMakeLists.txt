find_package(Protobuf REQUIRED)

protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)
add_executable(onnx2ncnn onnx2ncnn.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
target_include_directories(onnx2ncnn
    PRIVATE
        ${Protobuf_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(onnx2ncnn PRIVATE cxx_std_11)
target_link_libraries(onnx2ncnn PRIVATE ${Protobuf_LIBRARIES})
