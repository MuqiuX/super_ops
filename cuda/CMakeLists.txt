cmake_minimum_required(VERSION 3.18)
project(cuda_add LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "52 61 75 86")

# Utils library
add_library(utils STATIC
    src/utils/utils.cu
)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(utils PUBLIC CUDA::cudart)

# Main executable
add_executable(main src/main.cu)
target_link_libraries(main PRIVATE utils CUDA::cudart)
