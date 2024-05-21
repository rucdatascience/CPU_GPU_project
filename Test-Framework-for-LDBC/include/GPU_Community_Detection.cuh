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
__global__ void LPA(int *global_space_for_label, int *in_out_ptr_gpu, int *labels_gpu, int *new_labels_gpu, int CD_GRAPHSIZE);
__global__ void parallel_sort_labels(int *in_out_ptr_gpu, int *labels_out, int CD_GRAPHSIZE);
__global__ void extract_labels(int *in_out_ptr_gpu, int *ins_ptr_gpu, int *outs_ptr_gpu, int *ins_neighbor_gpu, int *outs_neighbor_gpu, int *labels_gpu, int *labels_out, int CD_GRAPHSIZE);
void checkCudaError(cudaError_t err, const char* msg);
void checkDeviceProperties();
void get_size();

template <typename T>
void pre_set(graph_structure<T> & graph, int& GRAPHSIZE);
__global__ void init_global_space(int *global_space_for_label,int* global_space_for_label_count, int CD_M, int CD_SET_THREAD);
extern "C"
int Community_Detection(graph_structure<double> & graph, float* elapsedTime,vector<int> &ans);
