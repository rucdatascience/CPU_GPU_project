#ifndef WS_SSSP_H
#define WS_SSSP_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <csr_graph.hpp>

__device__ __forceinline__ double atomicMinDouble (double * addr, double value);

__global__ void Relax(int* offsets, int* edges, double* weights, double* dis, int* queue, int* queue_size, int* visited);
__global__ void CompactQueue(int V, int* next_queue, int* next_queue_size, int* visited);
void gpu_shortest_paths(CSR_graph<double>& input_graph, int source, std::vector<double>& distance, double max_dis = 10000000000);
std::map<long long int, double> getGPUSSSP(graph_structure<double> & graph, CSR_graph<double> & csr_graph);
std::vector<std::string> gpu_shortest_paths_v2(graph_structure<double> & graph, CSR_graph<double> &csr_graph);

std::vector<std::pair<std::string, double>> Cuda_SSSP(graph_structure<double>& graph, CSR_graph<double>& csr_graph, std::string src_v, double max_dis = 10000000000);

#endif