#ifndef WS_SSSP_H
#define WS_SSSP_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "./graph_structure/graph_structure.h"

__global__ void Relax(int* offsets, int* edges, int* weights, int* dis, int* queue, int* queue_size, int* visited);
__global__ void CompactQueue(int V, int* next_queue, int* next_queue_size, int* visited);
void Workfront_Sweep(CSR_graph<int>& input_graph, int source, std::vector<int>& distance, int max_dis = 1000000);

#endif