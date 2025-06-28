// PageRank_update.cuh
#ifndef PAGERANK_CUH_
#define PAGERANK_CUH_

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <GPU_gpma/GPU_gpma.hpp>

using namespace std;

// Constants
#define SMALL_BOUND 6
#define NORMAL_BOUND 96
#define THREAD_PER_BLOCK 512

// Function prototypes

// CUDA kernels
__device__ double _atomicAdd(double* address, double val);
__global__ void importance(double *npr, double *pr, double damp, KEY_TYPE *keys_in, VALUE_TYPE *values_in, SIZE_TYPE *row_offset_in, int GRAPHSIZE);
__global__ void importance_v2(double *npr, double *pr, double damp, KEY_TYPE *keys_in, VALUE_TYPE *values_in, SIZE_TYPE *row_offset_in, int GRAPHSIZE);
__global__ void calculate_sink(double *pr, int *N_out_zero_gpu, int out_zero_size, double *sink_sum);
__global__ void initialization(double *pr, double *outs, KEY_TYPE *keys_out, VALUE_TYPE *values_out, SIZE_TYPE *row_offset_out, int N);
__global__ void Antecedent_division(double *pr,double *npr, double *outs,double redi_tele, int N);

void GPU_PR(graph_structure<double> &graph, GPMA& gpma_graph_in, GPMA &gpma_graph_out, vector<double> &result, int iterations, double damping);

std::vector<std::pair<std::string, double>> Cuda_PR(graph_structure<double> &graph, GPMA& gpma_graph_in, GPMA &gpma_graph_out, int iterations, double damping);

// The gpu version of the pagerank algorithm
// return the pagerank of each vertex based on the graph, damping factor and number of iterations.
// the pagerank value is stored in the results
void GPU_PR (graph_structure<double> &graph, GPMA& gpma_graph_in, GPMA &gpma_graph_out, vector<double>& result, int iterations, double damping) {
    int N = graph.V; // number of vertices in the graph
    int E = gpma_graph_in.get_size();
    double teleport = (1 - damping) / N; // teleport mechanism

    int* sink_vertex_gpu = nullptr;
    double* sink_sum = nullptr;
    double* pr = nullptr;
    double* npr = nullptr;
    double* outs = nullptr;

    dim3 blockPerGrid, threadPerGrid;

    vector<int> sink_vertexs;

    cudaMallocManaged(&outs, N * sizeof(double));
    cudaMallocManaged(&sink_sum, sizeof(double));
    cudaMallocManaged(&npr, N * sizeof(double));
    cudaMallocManaged(&pr, N * sizeof(double));

    cudaDeviceSynchronize(); // synchronize, ensure the cudaMalloc is complete

    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) { // use the cudaGetLastError to check for possible cudaMalloc errors
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    for (int i = 0; i < N; i++) { // traverse all vertices
        if (graph.OUTs[i].size()==0) // this means that the vertex has no edges
            sink_vertexs.push_back(i); // record the sink vertices
    }
    int out_zero_size = sink_vertexs.size(); // the number of sink vertices
    cudaMallocManaged(&sink_vertex_gpu, sink_vertexs.size() * sizeof(int));
    cudaDeviceSynchronize();
    cudaMemcpy(sink_vertex_gpu, sink_vertexs.data(), sink_vertexs.size() * sizeof(int), cudaMemcpyHostToDevice);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }
    
    blockPerGrid.x = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK; // the number of blocks used in the gpu
    // blockPerGrid_E.x = (E + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK; // the number of blocks used in the gpu
    threadPerGrid.x = THREAD_PER_BLOCK; // the number of threads used in the gpu

    int iteration = 0; // number of iterations

    initialization<<<blockPerGrid, threadPerGrid>>>(pr, outs, RAW_PTR(gpma_graph_out.keys), RAW_PTR(gpma_graph_out.values), RAW_PTR(gpma_graph_out.row_offset), N); // initializes the pagerank and calculates the reciprocal of the out-degree
    cudaDeviceSynchronize();
    while (iteration < iterations) { // continue for a fixed number of iterations
        *sink_sum = 0;
        calculate_sink<<<blockPerGrid, threadPerGrid, THREAD_PER_BLOCK * sizeof(double)>>>(pr, sink_vertex_gpu, out_zero_size, sink_sum); // calculate the sinksum
        cudaDeviceSynchronize();
        *sink_sum = (*sink_sum) * damping / N; // the redistributed value of sink vertices
        Antecedent_division<<<blockPerGrid, threadPerGrid>>>(pr, npr, outs, teleport + (*sink_sum), N);
        
        cudaDeviceSynchronize();
        importance<<<blockPerGrid, threadPerGrid>>>(npr, pr, damping, RAW_PTR(gpma_graph_in.keys), RAW_PTR(gpma_graph_in.values), RAW_PTR(gpma_graph_in.row_offset), N); // calculate importance
        // importance_v2<<<blockPerGrid_E, threadPerGrid>>>(npr, pr, damping, RAW_PTR(gpma_graph_in.keys), RAW_PTR(gpma_graph_in.values), RAW_PTR(gpma_graph_in.row_offset), E); // calculate importance
        cudaDeviceSynchronize();

        std::swap(pr, npr); // store the updated pagerank in the rank
        iteration++;
    }
    result.resize(N);
    cudaMemcpy(result.data(), pr, N * sizeof(double), cudaMemcpyDeviceToHost); // get gpu PR algorithm result
    cudaDeviceSynchronize();
    
    cudaFree(pr); // free memory
    cudaFree(npr);
    cudaFree(outs);
    cudaFree(sink_vertex_gpu);
    cudaFree(sink_sum);
}

// initialization of the pagerank state
__global__ void initialization(double *pr, double *outs, KEY_TYPE *keys_out, VALUE_TYPE *values_out, SIZE_TYPE *row_offset_out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    if (tid >= 0 && tid < N) {
        pr[tid] = 1 / (double)N; // the initial pagerank is 1/N
        SIZE_TYPE start = row_offset_out[tid];
        SIZE_TYPE end = row_offset_out[tid + 1];
        int deg = 0;
        for (int c = start; c < end; c ++) {
            KEY_TYPE new_v = keys_out[c] & 0xFFFFFFFF;
            VALUE_TYPE new_w = values_out[c];
            if (new_v != COL_IDX_NONE && new_w != VALUE_NONE) {
                deg ++;
            }
        }
        if (deg){ // determine whether the vertex has out-edge
            outs[tid] = 1 / ((double)deg); // calculate the reciprocal of the out-degree of each vertex to facilitate subsequent calculations
        } else {
            outs[tid] = 0; // consider importance value to be 0 for sink vertices
        }
    }
}

// compute division in advance, pr(u) / Nout(u), which is used to calculate the importance value
__global__ void Antecedent_division(double *pr, double *npr, double *outs, double redi_tele, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    if (tid >= 0 && tid < N) {
        pr[tid] *= outs[tid];
        npr[tid] = redi_tele; // the sum of redistributed value and teleport value
    }
}

// calculate importance
__global__ void importance(double *npr, double *pr, double damp, KEY_TYPE *keys_in, VALUE_TYPE *values_in, SIZE_TYPE *row_offset_in, int GRAPHSIZE) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    
    if (tid >= 0 && tid < GRAPHSIZE) {
        // begin and end of in edges
        double acc = 0; // sum of u belongs to Nin(v)
        SIZE_TYPE start = row_offset_in[tid];
        SIZE_TYPE end = row_offset_in[tid + 1];
        for (int c = start; c < end; c ++) {
            SIZE_TYPE new_v = SIZE_TYPE(keys_in[c] & 0xFFFFFFFF);
            VALUE_TYPE new_w = values_in[c];
            if (new_v != COL_IDX_NONE && new_w != VALUE_NONE) {
                acc += pr[new_v]; // val_col[c] is neighbor,rank get PR(u) row_value is denominator i.e. Nout
            }
        }
        npr[tid] += acc * damp; // scaling is damping factor
    }
    return;
}

// calculate importance
__global__ void importance_v2 (double *npr, double *pr, double damp, KEY_TYPE *keys_in, VALUE_TYPE *values_in, SIZE_TYPE *row_offset_in, int GRAPHSIZE) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    
    if (tid >= 0 && tid < GRAPHSIZE) {
        // begin and end of in edges
        SIZE_TYPE from_v = SIZE_TYPE(keys_in[tid] >> 32);
        SIZE_TYPE new_v = SIZE_TYPE(keys_in[tid] & COL_IDX_NONE);
        VALUE_TYPE new_w = values_in[tid];
        if (from_v != COL_IDX_NONE && new_v != COL_IDX_NONE && new_w != VALUE_NONE) {
            _atomicAdd(&npr[from_v], (pr[new_v] * damp));
        }
    }
    return;
}

// A reduction pattern was used to sum up the sink value
__global__ void calculate_sink(double *pr, int *N_out_zero_gpu, int out_zero_size, double *sink_sum) {
    extern __shared__ double sink[]; // Declare shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stid = threadIdx.x;

    if (tid < out_zero_size)
        sink[stid] = pr[N_out_zero_gpu[tid]]; // get PR(w)
    else
        sink[stid] = 0; // not the out-degree zero vertex
    __syncthreads(); // wait unitl finish Loading data into shared memory

    for (int i = blockDim.x / 2; i > 0; i >>= 1) { // get the sum of sink by reducing kernel function
        if (stid < i)
            sink[stid] += sink[stid + i];
        __syncthreads(); // Synchronize again to ensure that each step of the reduction operation is completed
    }
    if (stid == 0)
        _atomicAdd(sink_sum, sink[0]); // Write the result of each thread block into the output array
}

// Implementing atomic operations,
// that is, ensuring that adding operations to a specific
// memory location in a multi-threaded environment are thread safe.
__device__ double _atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// PageRank Algorithm on GPU
// return the pagerank of each vertex based on the graph, damping factor and number of iterations.
// the type of the vertex and pagerank are string
std::vector<std::pair<std::string, double>> Cuda_PR(graph_structure<double> &graph, GPMA& gpma_graph_in, GPMA &gpma_graph_out, int iterations, double damping){
    std::vector<double> result;
    GPU_PR(graph, gpma_graph_in, gpma_graph_out, result, iterations, damping); // get the pagerank in double type
    return graph.res_trans_id_val(result); // return the results in string type
}

#endif // PAGERANK_CUH_
