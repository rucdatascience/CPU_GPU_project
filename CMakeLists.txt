cmake_minimum_required(VERSION 3.9)
project(Test LANGUAGES CXX)

option(BUILD_CPU "Build the CPU version" ON)
option(BUILD_GPU "Build the GPU version" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

if(BUILD_CPU)
    set(EXECUTABLE_NAME_CPU "${PROJECT_NAME}_CPU")
    set(EXECUTABLE_OUTPUT_PATH_CPU ${PROJECT_SOURCE_DIR}/build/bin_cpu)

    add_executable(${EXECUTABLE_NAME_CPU} src/LDBC/LDBC_CPU_adj_list.cpp)

    target_include_directories(${EXECUTABLE_NAME_CPU} PUBLIC include)

    set_target_properties(${EXECUTABLE_NAME_CPU} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_CPU})
endif()

if(BUILD_GPU)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)

    set(EXECUTABLE_NAME_GPU "${PROJECT_NAME}_GPU")
    set(EXECUTABLE_OUTPUT_PATH_GPU ${PROJECT_SOURCE_DIR}/build/bin_gpu)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

    add_executable(${EXECUTABLE_NAME_GPU} src/LDBC/LDBC_GPU_csr.cu)

    target_include_directories(${EXECUTABLE_NAME_GPU} PUBLIC ${CUDA_INCLUDE_DIRS})

    target_link_libraries(${EXECUTABLE_NAME_GPU} ${CUDA_LIBRARIES})

    target_include_directories(${EXECUTABLE_NAME_GPU} PUBLIC include)

    set_target_properties(${EXECUTABLE_NAME_GPU} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_GPU})
    set_target_properties(${EXECUTABLE_NAME_GPU} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()