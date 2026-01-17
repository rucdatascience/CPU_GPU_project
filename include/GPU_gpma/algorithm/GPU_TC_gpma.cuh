#ifndef GPU_TC
#define GPU_TC
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

#define CALC_BLOCKS_NUM(threads, n) (((n) + (threads) - 1) / (threads))

__device__ void intersect_and_count(KEY_TYPE *keys, VALUE_TYPE *values, int start_u, int end_u, 
                                    int start_v, int end_v, int u, int v, int* per_node_triangles) {
    int i = start_u;
    int j = start_v;
    while (i < end_u && j < end_v) {
        KEY_TYPE keys_u = keys[i] & 0xFFFFFFFF;
        VALUE_TYPE values_u = values[i];
        while ((keys_u == COL_IDX_NONE || values_u == VALUE_NONE) && i < end_u) {
            i ++;
            keys_u = keys[i] & 0xFFFFFFFF, values_u = values[i];
        }

        KEY_TYPE keys_v = keys[j] & 0xFFFFFFFF;
        VALUE_TYPE values_v = values[j];
        while ((keys_v == COL_IDX_NONE || values_v == VALUE_NONE) && j < end_v) {
            j ++;
            keys_v = keys[j] & 0xFFFFFFFF, values_v = values[j];
        }

        if (i >= end_u || j >= end_v) {
            break;
        }

        if (keys_u == keys_v) {
            atomicAdd(&per_node_triangles[u], 1);
            atomicAdd(&per_node_triangles[v], 1);
            atomicAdd(&per_node_triangles[keys_u], 1);
            i ++; j ++;
        } else if (keys_u < keys_v) {
            i ++;
        } else {
            j ++;
        }
    }
}

__global__ void triangle_counting_per_node_kernel_opt (KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offset, 
                                                       int* __restrict__ per_node_triangles, int V) {
    int u = blockIdx.x;
    if (u >= V) return;
    int start_u = row_offset[u], end_u = row_offset[u + 1];
    for (int idx = threadIdx.x; idx < (end_u - start_u); idx += blockDim.x) {
        KEY_TYPE new_v = keys[start_u + idx] & 0xFFFFFFFF;
        VALUE_TYPE new_w = values[start_u + idx];
        if (new_v != COL_IDX_NONE && new_w != VALUE_NONE) {
            int v = new_v;
            int start_v = row_offset[v], end_v = row_offset[v + 1];
            intersect_and_count(keys, values, start_u, end_u, start_v, end_v, u, v, per_node_triangles);
        }
    }
}

__host__ std::vector<int> Cuda_TC_PerNode(GPMA& gpma_graph, int V) {
    const int THREADS_PER_BLOCK = 256;
    std::vector<int> h_per_node_counts(V, 0);

    int* d_per_node_counts;
    cudaMalloc(&d_per_node_counts, V * sizeof(int));
    cudaMemset(d_per_node_counts, 0, V * sizeof(int));
    cudaDeviceSynchronize();
    int BLOCKS_NUM = V;

    triangle_counting_per_node_kernel_opt<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
        RAW_PTR(gpma_graph.keys), RAW_PTR(gpma_graph.values), RAW_PTR(gpma_graph.row_offset), d_per_node_counts, V);
    cudaDeviceSynchronize();
    cudaMemcpy(h_per_node_counts.data(), d_per_node_counts, V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_per_node_counts);
    return h_per_node_counts;
}

inline std::vector<std::pair<std::string, int>> Cuda_TC(graph_structure<double> &graph, GPMA& gpma_graph) {
    int N = graph.size();
    std::vector<int> tcVecGPU = Cuda_TC_PerNode(gpma_graph, N);
    return graph.res_trans_id_val(tcVecGPU);
}

#endif