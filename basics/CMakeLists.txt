cmake_minimum_required(VERSION 3.12)

project(CRHPCS_basics LANGUAGES C CXX CUDA)

find_package(Kokkos REQUIRED)


add_subdirectory(cpu_vs_gpu/)
add_subdirectory(1.initialization/)
add_subdirectory(4.lambda_correct/)
add_subdirectory(5.views/)
add_subdirectory(heat_simulation/)
add_subdirectory(matmul/)
add_subdirectory(pi_leibniz/)
add_subdirectory(k_means/)
if(NOT Kokkos_ENABLE_CUDA)
message("Generating CPU only examples...")
  add_subdirectory(2.functors/)
  add_subdirectory(3.lambda/)
else()
message("Skipping CPU only examples...")
endif()
