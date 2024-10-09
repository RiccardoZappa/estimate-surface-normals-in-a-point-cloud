cmake_minimum_required(VERSION 3.18)  # Increased for better CUDA support

# Set the project name and version
project(HelloWorldCUDA LANGUAGES CXX CUDA VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the CUDA standard (optional, defaults to C++14 for CUDA 11+)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Find CUDA package
find_package(CUDA REQUIRED)
# Find PCL and OpenCV
find_package(PCL 1.8 REQUIRED) # Modify the version if needed
find_package(OpenCV REQUIRED)

set(CUDA_LIBRARIES ${CUDA_LIBRARIES} cusolver cublas)
# Add executable target
add_executable(hello_world_cuda kernel.cu)
add_executable(nearest_neighbours_cuda cuda_nearest_neighbours.cu)
add_executable(normal_estimation normalEstimation.cpp)

# Set CUDA architectures (adjust based on your target GPUs)
set_target_properties(hello_world_cuda PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86")

# Include CUDA headers
target_include_directories(hello_world_cuda PRIVATE ${CUDA_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

target_include_directories(nearest_neighbours_cuda PRIVATE ${PCL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

target_include_directories(normal_estimation PRIVATE  ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# Link against CUDA libraries
target_link_libraries(hello_world_cuda PRIVATE ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

target_link_libraries(nearest_neighbours_cuda PRIVATE ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

target_link_libraries(normal_estimation PRIVATE ${PCL_LIBRARIES} ${OpenCV_LIBS})

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