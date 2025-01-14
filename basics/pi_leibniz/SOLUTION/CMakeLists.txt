cmake_minimum_required(VERSION 3.12)

project(pi_leibniz LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(pi_leibniz main.cpp)

target_link_libraries(pi_leibniz Kokkos::kokkos)
