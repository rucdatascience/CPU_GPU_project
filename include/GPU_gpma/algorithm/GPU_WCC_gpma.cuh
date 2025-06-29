#ifndef WCCG
#define WCCG
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <vector>
#include <string.h>
#include <GPU_csr/GPU_csr.hpp>
#include <GPU_gpma/GPU_gpma.hpp>
using namespace std;
#define WCCG_THREAD_PER_BLOCK 512

__global__ void parent_init(int *parent, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    if (tid < N) { // tid decides process which vertex
        parent[tid] = tid; // each vertex is initially labeled by itself
    }
}

__global__ void compress(int *parent, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    if (tid < N) { // tid decides process which vertex
        while (parent[tid] != parent[parent[tid]]) {
            parent[tid] = parent[parent[tid]];
        }
    }
}

__global__ void get_freq(int *parent, int *freq, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    if (tid < N) // tid decides process which vertex
        atomicAdd(&freq[parent[tid]], 1);
}

__global__ void sampling(KEY_TYPE *keys_in, VALUE_TYPE *values_in, SIZE_TYPE *row_offset_in, int *parent, int N, int neighbor_round) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    if (tid < N) { // tid decides process which vertex
        SIZE_TYPE start = row_offset_in[tid];
        SIZE_TYPE end = row_offset_in[tid + 1];

        int i = start + neighbor_round;
        if (i < end) {
            KEY_TYPE new_v = keys_in[i] & 0xFFFFFFFF;
            VALUE_TYPE new_w = values_in[i];
            if (new_v != COL_IDX_NONE && new_w != VALUE_NONE) {
                int p1 = parent[tid], p2 = parent[new_v];
                while (p1 != p2) { // link
                    int h = p1 > p2 ? p1 : p2;
                    int l = p2 >= p1 ? p1 : p2;
                    int check = atomicCAS(&parent[h], h, l);
                    if (check == h) {
                        break;
                    }
                    p1 = parent[parent[h]];
                    p2 = parent[l];
                }
            }
        }
    }
}

__global__ void full_link(KEY_TYPE *keys_in, VALUE_TYPE *values_in, SIZE_TYPE *row_offset_in, int *parent, int most, int N, int neighbor_round) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    if (tid < N) { // tid decides process which vertex
        if (parent[tid] == most) {
            return;
        }
        SIZE_TYPE start = row_offset_in[tid];
        SIZE_TYPE end = row_offset_in[tid + 1];
        for (int i = start + neighbor_round; i < end; i++) {
            KEY_TYPE new_v = keys_in[i] & 0xFFFFFFFF;
            VALUE_TYPE new_w = values_in[i];
            if (new_v != COL_IDX_NONE && new_w != VALUE_NONE) {
                int p1 = parent[tid], p2 = parent[new_v];
                while (p1 != p2) { // link
                    int h = p1 > p2 ? p1 : p2;
                    int l = p2 >= p1 ? p1 : p2;
                    int check = atomicCAS(&parent[h], h, l);
                    if (check == h) {
                        break;
                    }
                    p1 = parent[parent[h]];
                    p2 = parent[l];
                }
            }
        }
    }
}

std::vector<int> WCC_GPU(graph_structure<double> &graph, GPMA& gpma_graph) {
    int N = graph.size(); // number of vertices in the graph

    dim3 init_label_block((N + WCCG_THREAD_PER_BLOCK - 1) / WCCG_THREAD_PER_BLOCK, 1, 1); // the number of blocks used in the gpu
    dim3 init_label_thread(WCCG_THREAD_PER_BLOCK, 1, 1);                                  // the number of threads used in the gpu

    int *parent = nullptr;
    int *freq = nullptr;
    cudaMallocManaged((void **)&parent, N * sizeof(int));
    cudaMallocManaged((void **)&freq, N * sizeof(int));
    cudaMemset(freq, 0, N * sizeof(int));
    parent_init<<<init_label_block, init_label_thread>>>(parent, N);
    cudaDeviceSynchronize();
    int it = 0, ITERATION = 2; // number of iterations
    while (it < ITERATION) {    // continue for a fixed number of iterations
        sampling<<<init_label_block, init_label_thread>>>(RAW_PTR(gpma_graph.keys), RAW_PTR(gpma_graph.values), RAW_PTR(gpma_graph.row_offset), parent, N, it);
        cudaDeviceSynchronize();
        compress<<<init_label_block, init_label_thread>>>(parent, N);
        cudaDeviceSynchronize();
        it++;
    }
    get_freq<<<init_label_block, init_label_thread>>>(parent, freq, N);
    int *c = thrust::max_element(thrust::device, freq, freq + N);
    int most_f_element = *c;
    full_link<<<init_label_block, init_label_thread>>>(RAW_PTR(gpma_graph.keys), RAW_PTR(gpma_graph.values), RAW_PTR(gpma_graph.row_offset), parent, most_f_element, N, ITERATION);
    cudaDeviceSynchronize();
    compress<<<init_label_block, init_label_thread>>>(parent, N);
    cudaDeviceSynchronize();
    
    std::vector<int> result(N);
    cudaMemcpy(result.data(), parent, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(parent);
    cudaFree(freq);
    return result;
}

std::vector<std::pair<std::string, std::string>> Cuda_WCC(graph_structure<double> &graph, GPMA& gpma_graph) {
    std::vector<int> wccVecGPU = WCC_GPU(graph, gpma_graph);
    return graph.res_trans_id_id(wccVecGPU);
}

#endif