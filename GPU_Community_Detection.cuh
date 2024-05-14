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

using namespace std;
#define CD_THREAD_PER_BLOCK 1024

__global__ void init_label(int* labels_gpu,int GRAPHSIZE);
__global__ void LPA(int* row_ptr_gpu, int* labels_gpu, int* neighbor_gpu, int* reduce_label, int* reduce_label_count,int GRAPHSIZE,int BLOCK_PER_VER);
__global__ void Updating_label(int* reduce_label, int* reduce_label_count, int* updating, int* labels_gpu,int GRAPHSIZE,int BLOCK_PER_VER);

template <typename T>
void make_csr(graph_structure<T> & graph, int& GRAPHSIZE);

extern "C"
int Community_Detection(graph_structure<double> & graph, float* elapsedTime);
