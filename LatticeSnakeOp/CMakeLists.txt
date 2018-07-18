cmake_minimum_required(VERSION 3.10)
project(LatticeSnakeOp)

set(CMAKE_CXX_STANDARD 11)

set(TENSORFLOW_INCLUDE /home/alex/Programs/anaconda3/lib/python3.6/site-packages/tensorflow/include)
include_directories(${TENSORFLOW_INCLUDE})

add_executable(LatticeSnakeOp LatticeSnake.cc LatticeSnake.h)
