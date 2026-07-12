cmake_minimum_required (VERSION 3.18)
project (cuint LANGUAGES CXX CUDA)
set(cuint_VERSION 1.0.0)
set(cuint_SOVERSION 1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Device architectures. Priority:
#   1. an explicit CMAKE_CUDA_ARCHITECTURES from the consumer (preferred knob),
#   2. the legacy CUDA_ARCHITECTURES variable,
#   3. a default covering up to sm_120 (PTX) for the detected CUDA toolkit
#      (CUDA 13 dropped sm_70 and below).
if(NOT CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;90-real;100-real;110-real;120-virtual")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real;100-real;101-real;120-virtual")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-virtual")
  endif()
endif()
message(STATUS "cuint CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

include(GNUInstallDirs)

set(MAX_L 4 CACHE STRING "The maximum angular momentum limit")
configure_file(
    "${PROJECT_SOURCE_DIR}/include/config.h.in"
    "${PROJECT_BINARY_DIR}/config.h"
)

set(cuintSrc
  src/overlap.cu src/dipole.cu src/quadrupole.cu
  src/pbc_overlap.cu
)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

add_library(cuint ${cuintSrc})
set_target_properties(cuint PROPERTIES
  VERSION ${cuint_VERSION}
  SOVERSION ${cuint_SOVERSION}
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)

target_compile_options(cuint PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_include_directories(cuint
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}
)

install(TARGETS cuint
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT lib
)

install(FILES include/cuint.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT dev
)

