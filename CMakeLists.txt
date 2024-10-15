cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app LANGUAGES C CXX)

find_package(Python)
if(Python_FOUND)
	if(EXISTS ${Python_SITELIB}/torch)
		list(APPEND CMAKE_PREFIX_PATH ${Python_SITELIB}/torch)
	endif()
endif()
find_package(Torch REQUIRED) 



add_executable(example-app example-app.cpp)
#target_compile_features(example-app PRIVATE cxx_std_20)
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
target_link_libraries(example-app torch)
