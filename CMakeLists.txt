cmake_minimum_required (VERSION 3.18)
project (cuint LANGUAGES CXX CUDA)
set(cuint_VERSION 1.0.0)
set(cuint_SOVERSION 1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
else()
  set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real")
endif()

include(GNUInstallDirs)

set(cuintSrc
  src/overlap.cu src/dipole.cu src/quadrupole.cu
)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

add_library(cuint ${cuintSrc})
set_target_properties(cuint PROPERTIES
  VERSION ${cuint_VERSION}
  SOVERSION ${cuint_SOVERSION}
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)

target_include_directories(cuint
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

install(TARGETS cuint
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT lib
)

install(FILES include/cuint.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT dev
)

