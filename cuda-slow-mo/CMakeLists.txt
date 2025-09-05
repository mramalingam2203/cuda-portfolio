cmake_minimum_required(VERSION 3.12)
project(cuda_slowmo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(video_interp
    src/main.cpp
    src/kernels.cu
)

target_link_libraries(video_interp PRIVATE ${OpenCV_LIBS})
set_target_properties(video_interp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
