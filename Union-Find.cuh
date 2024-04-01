#ifndef UF_CUH
#define UF_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <graph_v_of_v/graph_v_of_v.h>

template <typename T>
std::vector<std::vector<int>> gpu_connected_components(ARRAY_graph<T>& input_graph);

__device__ int findRoot(int* parent, int i);
__global__ void Hook(int* parent, int* Start_v, int* End_v, int E);

#endif