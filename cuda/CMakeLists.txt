cmake_minimum_required(VERSION 3.18)
project(cuda_add LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)


set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")


# Utils library
add_library(utils STATIC
    src/utils.cu
    src/accuracy_utils.cu
    src/ops/add.cu
    src/ops/sum.cu
)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(utils PUBLIC CUDA::cudart)


# cli
add_executable(prop src/cli/prop.cu)
target_link_libraries(prop PRIVATE utils CUDA::cudart)

# Main executable
# add_executable(main src/main.cu)
# target_link_libraries(main PRIVATE utils CUDA::cudart)

#add_executable(parallelism_performance tests/parallelism_performance.cu)
#target_link_libraries(parallelism_performance PRIVATE utils CUDA::cudart)

# -------------------------------
# course
# -------------------------------

# course001
add_executable(course001 tests/course001.cu)
target_link_libraries(course001 PRIVATE utils CUDA::cudart)

# course002
add_executable(course002 tests/course002.cu)
target_link_libraries(course002 PRIVATE utils CUDA::cudart)