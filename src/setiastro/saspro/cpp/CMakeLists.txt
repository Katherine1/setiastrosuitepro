cmake_minimum_required(VERSION 3.15)
project(saspro_cpp LANGUAGES CXX)

find_package(pybind11 REQUIRED CONFIG)

# Find OpenCV - user must have it installed via vcpkg, brew, apt, or set OpenCV_DIR
find_package(OpenCV REQUIRED)

# Include the current directory for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

pybind11_add_module(saspro_cpp MODULE bindings.cpp stacking.cpp image_ops.cpp deconvolver.cpp background_extractor.cpp live_stacker.cpp)

# Optimize for speed and fix Windows Macro conflicts
if(MSVC)
    # /DNOMINMAX and /DWIN32_LEAN_AND_MEAN are crucial to prevent Windows.h from polluting the namespace
    target_compile_options(saspro_cpp PRIVATE /O2 /Ob2 /Oi /Ot /Oy /GL /openmp /DNOMINMAX /DWIN32_LEAN_AND_MEAN /EHsc)
    target_link_options(saspro_cpp PRIVATE /LTCG)
elseif(APPLE)
    # macOS with Clang
    target_compile_options(saspro_cpp PRIVATE -O3 -Xpreprocessor -fopenmp)
    find_library(OMP_LIB omp HINTS /opt/homebrew/lib /usr/local/lib)
    if(OMP_LIB)
        target_link_libraries(saspro_cpp PRIVATE ${OMP_LIB})
    endif()
else()
    # Linux with GCC
    target_compile_options(saspro_cpp PRIVATE -O3 -march=native -fopenmp)
    target_link_options(saspro_cpp PRIVATE -fopenmp)
endif()

target_link_libraries(saspro_cpp PRIVATE ${OpenCV_LIBS})

# C++17 or later
target_compile_features(saspro_cpp PRIVATE cxx_std_17)

install(TARGETS saspro_cpp DESTINATION setiastro/saspro/cpp)
