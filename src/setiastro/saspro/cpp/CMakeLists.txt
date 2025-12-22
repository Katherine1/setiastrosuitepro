cmake_minimum_required(VERSION 3.15)
project(saspro_cpp LANGUAGES CXX)

find_package(pybind11 REQUIRED CONFIG)
find_package(OpenCV REQUIRED)

# Include the current directory for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(saspro_cpp MODULE bindings.cpp stacking.cpp)

# Optimize for speed
if(MSVC)
    target_compile_options(saspro_cpp PRIVATE /O2 /Ob2 /Oi /Ot /Oy /GL /openmp)
    target_link_options(saspro_cpp PRIVATE /LTCG)
else()
    target_compile_options(saspro_cpp PRIVATE -O3 -march=native -fopenmp)
    target_link_options(saspro_cpp PRIVATE -fopenmp)
endif()

target_link_libraries(saspro_cpp PRIVATE ${OpenCV_LIBS})

# C++17 or later
target_compile_features(saspro_cpp PRIVATE cxx_std_17)
