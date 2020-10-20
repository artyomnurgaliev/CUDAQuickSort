cmake_minimum_required(VERSION 3.17)
project(CUDAQuickSort CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDAQuickSort main.cu)

set_target_properties(
        CUDAQuickSort
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)