#ifndef GPU_BFS
#define GPU_BFS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <csr_graph.hpp>

#include <map>

__global__ void bfs_kernel(int* edges, int* start, int* visited, int* queue, int* next_queue, int* queue_size, int* next_queue_size, int max_depth);

//template<typename T>
std::vector<int> cuda_bfs(CSR_graph<double>& input_graph, int source_vertex, int max_depth = INT_MAX);

std::vector<std::pair<std::string, int>> Cuda_Bfs(graph_structure<double>& graph, CSR_graph<double>& csr_graph, std::string src_v, int min_depth = 0, int max_depth = INT_MAX);

#endif