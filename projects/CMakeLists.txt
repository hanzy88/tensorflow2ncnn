#set(OpenCV_PATH /home/zyhan/ncnn/3rdparty/lib)
#find_package(OpenCV QUIET COMPONENTS 3rdparty)
#find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
#if(NOT OpenCV_FOUND)
#    #find_package(OpenCV REQUIRED COMPONENTS 3rdparty)
#    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
#endif()
set(OpenCV_LIBS /home/zyhan/ncnn/3rdparty/lib/libopencv_core.so /home/zyhan/ncnn/3rdparty/lib/libopencv_highgui.so /home/zyhan/ncnn/3rdparty/lib/libopencv_imgproc.so /home/zyhan/ncnn/3rdparty/lib/libopencv_imgcodecs.so)


include_directories(/home/zyhan/ncnn/3rdparty/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

set(NCNN_EXAMPLE_LINK_LIBRARIES ncnn ${OpenCV_LIBS})
if(NCNN_VULKAN)
    list(APPEND NCNN_EXAMPLE_LINK_LIBRARIES ${Vulkan_LIBRARY})
endif()


add_executable(myNet myNet.cpp)
target_link_libraries(myNet ${NCNN_EXAMPLE_LINK_LIBRARIES})

add_executable(yolo yolo.cpp)
target_link_libraries(yolo ${NCNN_EXAMPLE_LINK_LIBRARIES})
