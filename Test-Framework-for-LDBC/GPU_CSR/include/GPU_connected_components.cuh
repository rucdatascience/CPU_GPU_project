#ifndef UF_CUH
#define UF_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#include "csr_graph.hpp"

//template <typename T>
extern "C"
std::vector<std::vector<int>> gpu_connected_components(CSR_graph<double>& input_graph, float* elapsedTime, int threads = 1024000);

__device__ int findRoot(int* parent, int i);
__global__ void Hook(int* parent, int* Start_v, int* End_v, int E);
std::vector<std::vector<std::string>> getGPUWCC(LDBC<double> & graph, CSR_graph<double>& csr_graph);

#endif