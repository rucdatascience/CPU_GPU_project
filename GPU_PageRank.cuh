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
extern int ITERATION;
extern int ALPHA;
// External global variables
extern int GRAPHSIZE;
extern int *graphSize;
extern int *row_point, *val_col;
extern int *row_size;
extern double *row_value;
extern double *Rank, *diff_array, *reduce_array;
extern double *newRank, *F, *temp;

// Function prototypes
bool cmp(const std::vector<pair<int, int>>& a, const std::vector<pair<int, int>>& b);

double Method(double *rank, int &iteration);

// CUDA kernels
__global__ void add_scaling(double *newRank, double *oldRank, double scaling);
__global__ void tinySolve(double *newRank, double *rank, double scaling, int *row_point, int *row_size, double *row_value, int *val_col);
// __global__ void vec_diff(double *diff, double *newRank, double *oldRank);
// __global__ void reduce_kernel(double *input, double *output);

extern "C"
void PageRank(graph_structure<double> &graph, float* elapsedTime);

#endif // PAGERANK_CUH_
