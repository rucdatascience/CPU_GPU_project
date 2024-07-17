cmake_minimum_required(VERSION 3.9)
project(Test LANGUAGES CXX)

option(BUILD_CPU "Build the CPU version" ON)
option(BUILD_GPU "Build the GPU version" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories(include)

if(BUILD_CPU)
    set(EXECUTABLE_NAME "${PROJECT_NAME}_CPU")
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin_cpu)

    aux_source_directory(src_cpu SRC_CPU)
    add_executable(${EXECUTABLE_NAME} ${SRC_CPU})
    target_include_directories(${EXECUTABLE_NAME} PUBLIC include_cpu)
endif()

if(BUILD_GPU)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)

    set(EXECUTABLE_NAME "${PROJECT_NAME}_GPU")
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin_gpu)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

    aux_source_directory(src_gpu SRC_GPU)
    add_executable(${EXECUTABLE_NAME} ${SRC_GPU})
    target_include_directories(${EXECUTABLE_NAME} PUBLIC include_gpu)

    target_link_libraries(${EXECUTABLE_NAME} ${CUDA_LIBRARIES})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()