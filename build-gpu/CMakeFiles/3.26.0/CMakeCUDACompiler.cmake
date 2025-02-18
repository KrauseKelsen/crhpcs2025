set(CMAKE_CUDA_COMPILER "/opt/tools/cuda-11.6.2-gcc-9.3.0/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/opt/compilers/gcc-11.1.0/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.6.124")
set(CMAKE_CUDA_DEVICE_LINKER "/opt/tools/cuda-11.6.2-gcc-9.3.0/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/opt/tools/cuda-11.6.2-gcc-9.3.0/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "11.1")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/opt/tools/cuda-11.6.2-gcc-9.3.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/opt/tools/cuda-11.6.2-gcc-9.3.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.6.124")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/opt/tools/cuda-11.6.2-gcc-9.3.0")

set(CMAKE_CUDA_ARCHITECTURES_ALL "35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "35-real;50-real;60-real;70-real;80")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/opt/tools/cuda-11.6.2-gcc-9.3.0/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/opt/tools/cuda-11.6.2-gcc-9.3.0/targets/x86_64-linux/lib/stubs;/opt/tools/cuda-11.6.2-gcc-9.3.0/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/opt/tools/kokkos-cuda/install/kokkos/include;/opt/compilers/gcc-9.3.0/include;/opt/compilers/gcc-11.1.0/include/c++/11.1.0;/opt/compilers/gcc-11.1.0/include/c++/11.1.0/x86_64-pc-linux-gnu;/opt/compilers/gcc-11.1.0/include/c++/11.1.0/backward;/opt/compilers/gcc-11.1.0/lib/gcc/x86_64-pc-linux-gnu/11.1.0/include;/usr/local/include;/opt/compilers/gcc-11.1.0/include;/opt/compilers/gcc-11.1.0/lib/gcc/x86_64-pc-linux-gnu/11.1.0/include-fixed;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/opt/tools/cuda-11.6.2-gcc-9.3.0/targets/x86_64-linux/lib/stubs;/opt/tools/cuda-11.6.2-gcc-9.3.0/targets/x86_64-linux/lib;/opt/compilers/gcc-11.1.0/lib64;/opt/tools/kokkos-cuda/install/kokkos/lib64;/opt/compilers/gcc-9.3.0/lib64;/opt/compilers/gcc-11.1.0/lib/gcc/x86_64-pc-linux-gnu/11.1.0;/lib64;/usr/lib64;/opt/compilers/gcc-11.1.0/lib;/opt/compilers/gcc-9.3.0/lib;/opt/tools/cuda-11.6.2-gcc-9.3.0/lib64")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
