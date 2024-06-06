// PageRank.cuh
#ifndef PAGERANK_CUH_
#define PAGERANK_CUH_

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <graph_structure/graph_structure.hpp> // Assuming graph_structure.h is properly set up with its own header guards

using namespace std;

// Constants
#define SMALL_BOUND 6
#define NORMAL_BOUND 96
#define THREAD_PER_BLOCK 512


// Function prototypes
bool cmp(const std::vector<pair<int, int>>& a, const std::vector<pair<int, int>>& b);

double Method(double *rank, int &iteration);

// CUDA kernels
__device__ double _atomicAdd(double* address, double val);
__global__ void add_scaling(double *newRank, double *oldRank, double scaling, int GRAPH_SIZE);
__global__ void tinySolve(double *newRank, double *rank, double scaling, int *row_point, int *row_size, double *row_value, int *val_col, int GRAPH_SIZE);
__global__ void calculate_sink(double* rank,int* N_out_zero_gpu,int out_zero_size,double *sink_sum);
// __global__ void vec_diff(double *diff, double *newRank, double *oldRank);
// __global__ void reduce_kernel(double *input, double *output);

extern "C"
void gpu_PageRank(graph_structure<double> &graph, float* elapsedTime, vector<double> & res);

#endif // PAGERANK_CUH_
