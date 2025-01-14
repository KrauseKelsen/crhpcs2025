cmake_minimum_required(VERSION 3.12)

project(matmul LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(matmul main.cpp)

target_link_libraries(matmul Kokkos::kokkos)
