cmake_minimum_required(VERSION 3.18)
project(cuda_add LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)


set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")


# Utils library
add_library(utils STATIC
    src/utils/utils.cu
    src/utils/accuracy_utils.cu
    src/ops/add.cu
    src/ops/sum.cu
)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(utils PUBLIC CUDA::cudart)


# Main executable
add_executable(main src/main.cu)
target_link_libraries(main PRIVATE utils CUDA::cudart)


# Test executables
add_executable(test_add tests/test_add.cu)
target_link_libraries(test_add PRIVATE utils CUDA::cudart)

add_executable(test_sum tests/test_sum.cu)
target_link_libraries(test_sum PRIVATE utils CUDA::cudart)
