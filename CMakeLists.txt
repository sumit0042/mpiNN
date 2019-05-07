cmake_minimum_required(VERSION 3.14)
project(cudaCPU)

set(CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

include_directories(layers)

add_executable(cudaCPU
    layers/cross_entropy.cpp
    layers/layer.hpp
    layers/linear.cpp
    layers/relu.cpp
    layers/sigmoid.cpp
    layers/softmax.cpp
    ffn.cpp
    matrix.cpp
    test.cpp
    train.cpp
    utils.cpp utils.hpp)
