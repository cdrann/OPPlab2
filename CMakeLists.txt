cmake_minimum_required(VERSION 3.14)
project(OPPlab2)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

#add_executable(OPPlab2 ver0.cpp ver1.cpp ver2.cpp ver3.cpp)
add_executable(OPPlab2  ver1.cpp )