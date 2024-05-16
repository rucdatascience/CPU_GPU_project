#ifndef WS_SSSP_H
#define WS_SSSP_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <graph_structure/graph_structure.hpp>

__device__ __forceinline__ double atomicMinDouble (double * addr, double value);

__global__ void Relax(int* offsets, int* edges, double* weights, double* dis, int* queue, int* queue_size, int* visited);
__global__ void CompactQueue(int V, int* next_queue, int* next_queue_size, int* visited);
void Workfront_Sweep(CSR_graph<double>& input_graph, int source, std::vector<double>& distance, float* elapsedTime, double max_dis = 10000000000);

#endif