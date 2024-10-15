cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

# TODO: move this path to CMakePresets
set(CMAKE_PREFIX_PATH /home/mike/DSR/libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
#target_compile_features(example-app PRIVATE cxx_std_20)
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
