cmake_minimum_required(VERSION 3.12)

project(CRHPCS_basics LANGUAGES C CXX CUDA)

add_subdirectory(cpu_vs_gpu/)
add_subdirectory(initialization/)
add_subdirectory(matmul/)
add_subdirectory(pi_leibniz/)
