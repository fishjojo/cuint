cmake_minimum_required (VERSION 3.18)

project (cuint CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
else()
  set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real")
endif()

add_library(cuint SHARED
  src/overlap.cu
  src/dipole.cu	
  src/quadrupole.cu
)

set_target_properties(cuint PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
