cmake_minimum_required(VERSION 3.12)

project(cpu_vs_gpu LANGUAGES CUDA C)

set(CMAKE_C_STANDARD 99)

find_package(CUDA REQUIRED)

add_executable(cpu_version main.c)
add_executable(gpu_version main.cu)
