cmake_minimum_required(VERSION 3.10)
project(cuda_grayscale)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(grayscale src/main.cpp src/grayscale.cu src/colorspace.cu src/gamma_brightness.cu)
target_link_libraries(grayscale ${OpenCV_LIBS})
