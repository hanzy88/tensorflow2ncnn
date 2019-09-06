find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

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
