#ifndef GPU_BFS
#define GPU_BFS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <graph_structure/graph_structure.hpp>

__global__ void bfs_kernel(int* edges, int* start, int* visited, int* queue, int* next_queue, int* queue_size, int* next_queue_size, int max_depth);

//template<typename T>
extern "C" 
std::vector<int> cuda_bfs(CSR_graph<double>& input_graph, int source_vertex, float* elapsedTime, int max_depth = INT_MAX);

#endif