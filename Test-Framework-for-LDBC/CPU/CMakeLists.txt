cmake_minimum_required(VERSION 3.9)

project(test-cpu LANGUAGES CXX)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} DIR_SRCS)

add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_sources(test-cpu PRIVATE ${DIR_SRCS})
