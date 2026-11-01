cmake_minimum_required(VERSION 3.10)
project(zipvoice_nano LANGUAGES CXX)

# JetPack 4.x ships CUDA 10 and older CMake. Keep this optional target isolated
# from the Python package and from the modern TensorRT/Triton runtime.
find_package(CUDA 10.0 REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATH_SUFFIXES aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer)
find_library(TENSORRT_PARSER_LIBRARY nvonnxparser)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_PARSER_LIBRARY)
  message(FATAL_ERROR "TensorRT 6 headers, nvinfer and nvonnxparser are required")
endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
list(APPEND CUDA_NVCC_FLAGS "-O3" "-std=c++11"
     "-gencode=arch=compute_53,code=sm_53"
     "-gencode=arch=compute_53,code=compute_53")

cuda_add_library(zipvoice_nano SHARED
  src/runtime.cu src/register.cpp src/relative_copy.cu src/relative_4d.cu
  src/slice_copy.cu src/row_softmax.cu src/swoosh.cu src/fusions.cu src/istft.cu)
target_link_libraries(zipvoice_nano ${TENSORRT_LIBRARY} ${CUDA_CUFFT_LIBRARIES})
add_executable(build_decoder src/build_decoder.cpp)
target_link_libraries(build_decoder zipvoice_nano ${TENSORRT_PARSER_LIBRARY})
set_target_properties(build_decoder PROPERTIES BUILD_RPATH "$ORIGIN")

include(CTest)
if(BUILD_TESTING)
  cuda_add_executable(test_fusions tests/test_fusions.cu)
  target_link_libraries(test_fusions zipvoice_nano)
  cuda_add_executable(test_swoosh tests/test_swoosh.cu)
  target_link_libraries(test_swoosh zipvoice_nano)
  add_executable(test_relative tests/test_relative.cpp)
  target_link_libraries(test_relative zipvoice_nano ${CUDA_CUDART_LIBRARY})
  add_executable(test_slice tests/test_slice.cpp)
  target_link_libraries(test_slice zipvoice_nano ${CUDA_CUDART_LIBRARY})
  foreach(test fusions swoosh relative slice)
    add_test(NAME ${test} COMMAND test_${test})
  endforeach()
endif()
