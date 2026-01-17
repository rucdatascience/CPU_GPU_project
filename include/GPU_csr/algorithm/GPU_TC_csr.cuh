#pragma once

#include "cub/cub.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <vector>
#include <algorithm>

#include <GPU_gpma/GPU_gpma.hpp>
#include <GPU_csr/GPU_csr.hpp>

#define CALC_BLOCKS_NUM(threads, n) (((n) + (threads) - 1) / (threads))

__device__ void intersect_and_count(int* col_indices, int start_u, int end_u, int start_v, int end_v,
    int u, int v, int* per_node_triangles) {
    int i = start_u;
    int j = start_v;
    while (i < end_u && j < end_v) {
        int w_u = col_indices[i];
        int w_v = col_indices[j];
        if (w_u == w_v) {
            atomicAdd(&per_node_triangles[u], 1);
            atomicAdd(&per_node_triangles[v], 1);
            atomicAdd(&per_node_triangles[w_u], 1);
            i ++; j ++;
        } else if (w_u < w_v) {
            i ++;
        } else {
            j ++;
        }
    }
}

__global__ void triangle_counting_per_node_kernel_opt (int* __restrict__ row_offsets, int* __restrict__ col_indices, int* __restrict__ per_node_triangles, int node_size) {
    int u = blockIdx.x;
    if (u >= node_size) return;
    int start_u = row_offsets[u], end_u = row_offsets[u + 1];
    for (int idx = threadIdx.x; idx < (end_u - start_u); idx += blockDim.x) {
        int v = col_indices[start_u + idx];
        int start_v = row_offsets[v], end_v = row_offsets[v + 1];
        intersect_and_count(col_indices, start_u, end_u, start_v, end_v, u, v, per_node_triangles);
    }
}

__host__ std::vector<int> Cuda_TC_PerNode(int* d_row_offsets, int* d_col_indices, int node_size) {
    const int THREADS_PER_BLOCK = 256;
    std::vector<int> h_per_node_counts(node_size, 0);

    int* d_per_node_counts;
    cudaMalloc(&d_per_node_counts, node_size * sizeof(int));
    cudaMemset(d_per_node_counts, 0, node_size * sizeof(int));

    int BLOCKS_NUM = node_size;

    triangle_counting_per_node_kernel_opt<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
        d_row_offsets, d_col_indices, d_per_node_counts, node_size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_per_node_counts.data(), d_per_node_counts, node_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_per_node_counts);
    return h_per_node_counts;
}

inline std::vector<std::pair<std::string, int>> Cuda_TC (graph_structure<double> &graph, CSR_graph<double> &csr_graph) {
    int V = graph.V;
    std::vector<int> per_node_counts = Cuda_TC_PerNode(csr_graph.out_pointer, csr_graph.out_edge, V);
    return graph.res_trans_id_val(per_node_counts);
}