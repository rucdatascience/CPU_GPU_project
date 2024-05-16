#pragma once

#include "cuda_runtime.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <string.h>
#include <graph_structure/graph_structure.hpp>
#include <unistd.h>
using namespace std;
#define CD_THREAD_PER_BLOCK 1024

__global__ void init_label(int* labels_gpu,int CD_GRAPHSIZE);
__global__ void LPA(int* outs_ptr_gpu, int* labels_gpu, int* outs_neighbor_gpu, int* reduce_label, int* reduce_label_count,int CD_GRAPHSIZE,int BLOCK_PER_VER,int * ins_ptr_gpu,int* ins_neighbor_gpu,int epoch_it,int epoch_size);
__global__ void Updating_label(int* reduce_label, int* reduce_label_count,  int* labels_gpu,int CD_GRAPHSIZE,int BLOCK_PER_VER,int epoch_it,int epoch_size);
void checkCudaError(cudaError_t err, const char* msg);
void checkDeviceProperties();
template <typename T>
void make_csr(graph_structure<T> & graph, int& GRAPHSIZE);

extern "C"
int Community_Detection(graph_structure<double> & graph, float* elapsedTime);
