#ifndef UF_CUH
#define UF_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <csr_graph.hpp>

std::vector<int> gpu_connected_components(CSR_graph<double>& input_graph, int threads = 1024000);

__device__ int findRoot(int* parent, int i);
__global__ void Hook(int* parent, int* Start_v, int* End_v, int E);
//std::vector<std::vector<std::string>> gpu_connected_components_v2(CSR_graph<double>& csr_graph, float* elapsedTime);

std::vector<std::pair<std::string, std::string>> Cuda_WCC(graph_structure<double>& graph, CSR_graph<double>& csr_graph);

#endif