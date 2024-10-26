cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(prototype)

# NB: This dir is not in git. 
# Download it and unpack https://pytorch.org/cppdocs/installing.html
#set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/ext/libtorch)
set(CMAKE_PREFIX_PATH /opt/dev/libtorch)

#set(CMAKE_CXX_ABI_COMPILED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(prototype example-app.cpp)
target_compile_features(prototype PRIVATE cxx_std_20)
target_link_libraries(prototype "${TORCH_LIBRARIES}")
