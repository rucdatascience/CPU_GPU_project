// PageRank_update.cuh
#ifndef PAGERANK_CUH_
#define PAGERANK_CUH_

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include "../include/ldbc.hpp"

#include "csr_graph.hpp"
#include "ldbc.hpp"
using namespace std;

// Constants
#define SMALL_BOUND 6
#define NORMAL_BOUND 96
#define THREAD_PER_BLOCK 512


// Function prototypes

// CUDA kernels
__device__ double _atomicAdd(double* address, double val);
__global__ void importance(double *npr, double *pr,  double damp, int *in_edge, int *in_pointer, int GRAPHSIZE);
__global__ void calculate_sink(double *pr, int *N_out_zero_gpu, int out_zero_size, double *sink_sum);
__global__ void initialization(double *pr, double *outs, int *out_pointer, int N);
__global__ void calculate_acc(double *pr,int *in_edge, int begin,int end,double *acc);
__global__ void Antecedent_division(double *pr,double *npr, double *outs,double redi_tele, int N);
extern "C"
// void GPU_PR(graph_structure<double> &graph, float *elapsedTime, vector<double> &result,int *in_pointer, int *out_pointer,int *in_edge,int *out_edge);
void GPU_PR(LDBC<double> &graph, float *elapsedTime, vector<double> &result,int *in_pointer, int *out_pointer,int *in_edge,int *out_edge);
std::map<long long int, double> getGPUPR(LDBC<double> & graph, CSR_graph<double> & csr_graph);
std::vector<std::string> GPU_PR_v2(LDBC<double> & graph, CSR_graph<double> &csr_graph);
void GPU_PR_v3(LDBC<double> &graph,float *elapsedTime, std::vector<std::string> &result,int *in_pointer, int *out_pointer,int *in_edge,int *out_edge);
#endif // PAGERANK_CUH_
