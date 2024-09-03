cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(HelloWorld VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add executable target
add_executable(hello_world main.cpp)


# Check for C and C++ compilers
enable_language(C CXX)

if(NOT CMAKE_C_COMPILER)
    message(FATAL_ERROR "C compiler not found!")
endif()

if(NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "C++ compiler not found!")
endif()

# Print compiler information
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")