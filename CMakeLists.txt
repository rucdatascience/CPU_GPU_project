cmake_minimum_required(VERSION 3.9)
project(Test LANGUAGES CXX CUDA)

option(BUILD_CPU "Build the CPU version" ON)
option(BUILD_GPU_CSR "Build the GPU_CSR version" ON)
option(BUILD_GPU_GPMA "Build the GPU_GPMA version" ON)
option(BUILD_GPU_ADJ "Build the GPU_ADJ version" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

if(BUILD_CPU)
    set(EXECUTABLE_NAME_CPU "${PROJECT_NAME}_CPU")
    set(CPU_example "CPU_example")
    set(EXECUTABLE_OUTPUT_PATH_CPU ${PROJECT_SOURCE_DIR}/build/bin_cpu)

    add_executable(${EXECUTABLE_NAME_CPU} src/LDBC/LDBC_CPU_adj_list.cpp)
    add_executable(${CPU_example} src/CPU_adj_list/CPU_example.cpp)

    target_include_directories(${EXECUTABLE_NAME_CPU} PUBLIC include)
    target_include_directories(${CPU_example} PUBLIC include)

    set_target_properties(${EXECUTABLE_NAME_CPU} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_CPU})
    set_target_properties(${CPU_example} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_CPU})
endif()

if(BUILD_GPU_CSR)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    
    set(EXECUTABLE_NAME_GPU "${PROJECT_NAME}_GPU_CSR")
    set(GPU_example_csr "GPU_example_csr")
    set(EXECUTABLE_OUTPUT_PATH_GPU ${PROJECT_SOURCE_DIR}/build/bin_gpu)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

    add_executable(${EXECUTABLE_NAME_GPU} src/LDBC/LDBC_GPU_csr.cu)
    add_executable(${GPU_example_csr} src/GPU_csr/GPU_csr_example.cu)

    target_include_directories(${EXECUTABLE_NAME_GPU} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_include_directories(${GPU_example_csr} PUBLIC ${CUDA_INCLUDE_DIRS})

    target_link_libraries(${EXECUTABLE_NAME_GPU} ${CUDA_LIBRARIES})
    target_link_libraries(${GPU_example_csr} ${CUDA_LIBRARIES})

    target_include_directories(${EXECUTABLE_NAME_GPU} PUBLIC include)
    target_include_directories(${GPU_example_csr} PUBLIC include)

    set_target_properties(${EXECUTABLE_NAME_GPU} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_GPU})
    set_target_properties(${EXECUTABLE_NAME_GPU} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${GPU_example_csr} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_GPU})
    set_target_properties(${GPU_example_csr} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_GPU_GPMA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    
    set(EXECUTABLE_NAME_GPU "${PROJECT_NAME}_GPU_GPMA")
    set(GPU_example_gpma "GPU_example_gpma")
    set(EXECUTABLE_OUTPUT_PATH_GPU ${PROJECT_SOURCE_DIR}/build/bin_gpu)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

    add_executable(${EXECUTABLE_NAME_GPU} src/LDBC/LDBC_GPU_gpma.cu)
    add_executable(${GPU_example_gpma} src/GPU_gpma/GPU_gpma_example.cu)

    target_include_directories(${EXECUTABLE_NAME_GPU} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_include_directories(${GPU_example_gpma} PUBLIC ${CUDA_INCLUDE_DIRS})

    target_link_libraries(${EXECUTABLE_NAME_GPU} ${CUDA_LIBRARIES})
    target_link_libraries(${GPU_example_gpma} ${CUDA_LIBRARIES})

    target_include_directories(${EXECUTABLE_NAME_GPU} PUBLIC include)
    target_include_directories(${GPU_example_gpma} PUBLIC include)

    set_target_properties(${EXECUTABLE_NAME_GPU} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_GPU})
    set_target_properties(${EXECUTABLE_NAME_GPU} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${GPU_example_gpma} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_GPU})
    set_target_properties(${GPU_example_gpma} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_GPU_ADJ)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    
    set(EXECUTABLE_NAME_GPU "${PROJECT_NAME}_GPU_ADJ")
    set(GPU_example_adj "GPU_example_adj")
    set(EXECUTABLE_OUTPUT_PATH_GPU ${PROJECT_SOURCE_DIR}/build/bin_gpu)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

    add_executable(${EXECUTABLE_NAME_GPU} src/LDBC/LDBC_GPU_adj.cu)
    add_executable(${GPU_example_adj} src/GPU_adj_list/GPU_adj_example.cu)

    target_include_directories(${EXECUTABLE_NAME_GPU} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_include_directories(${GPU_example_adj} PUBLIC ${CUDA_INCLUDE_DIRS})

    target_link_libraries(${EXECUTABLE_NAME_GPU} ${CUDA_LIBRARIES})
    target_link_libraries(${GPU_example_adj} ${CUDA_LIBRARIES})

    target_include_directories(${EXECUTABLE_NAME_GPU} PUBLIC include)
    target_include_directories(${GPU_example_adj} PUBLIC include)

    set_target_properties(${EXECUTABLE_NAME_GPU} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_GPU})
    set_target_properties(${EXECUTABLE_NAME_GPU} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${GPU_example_adj} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH_GPU})
    set_target_properties(${GPU_example_adj} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()