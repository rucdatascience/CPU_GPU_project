#ifndef UF_CUH
#define UF_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "./graph_structure/graph_structure.h"

template <typename T>
std::vector<std::vector<int>> gpu_connected_components(CSR_graph<T>& input_graph);

__device__ int findRoot(int* parent, int i);
__global__ void Hook(int* parent, int* Start_v, int* End_v, int E);

#endif