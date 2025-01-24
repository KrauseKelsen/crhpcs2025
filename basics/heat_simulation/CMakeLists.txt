cmake_minimum_required(VERSION 3.12)

project(heat_simulation LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(heat_simulation main.cpp)

target_link_libraries(heat_simulation Kokkos::kokkos)