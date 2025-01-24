#ifndef GPU_WCC_ADJ
#define GPU_WCC_ADJ

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>

#include <GPU_adj_list/GPU_adj.hpp>

using namespace std;
#define WCCG_THREAD_PER_BLOCK 512

__global__ void parent_init(int *parent, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

    if (tid < N) // tid decides process which vertex
    {
        parent[tid] = tid; // each vertex is initially labeled by itself
    }
}

__global__ void compress(int *parent, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

    if (tid < N) // tid decides process which vertex
    {
        while (parent[tid] != parent[parent[tid]])
        {
            parent[tid] = parent[parent[tid]];
        }
    }
}

__global__ void get_freq(int *parent, int *freq, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

    if (tid < N) // tid decides process which vertex
        atomicAdd(&freq[parent[tid]], 1);
}

__global__ void sampling(cuda_vector<std::pair<int, double>>** all_edge, int *parent, int N, int neighbor_round)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    int u = tid;
    if (tid < N) // tid decides process which vertex
    {

        int i = neighbor_round;
        if (i < all_edge[u]->size())
        {
            int v = (*all_edge[u])[i].first;
            int p1 = parent[u], p2 = parent[v];
            while (p1 != p2)
            { // link
                int h = p1 > p2 ? p1 : p2;
                int l = p2 >= p1 ? p1 : p2;
                int check = atomicCAS(&parent[h], h, l);
                if (check == h)
                {
                    break;
                }
                p1 = parent[parent[h]];
                p2 = parent[l];
            }
        }
    }
}

__global__ void full_link(cuda_vector<std::pair<int, double>>** all_edge, int *parent, int most, int N, int neighbor_round)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    int u = tid;
    if (tid < N) // tid decides process which vertex
    {
        if (parent[tid] == most)
        {
            return;
        }

        int all_edge_size = all_edge[u]->size();
        for (int i = 0; i < all_edge_size; i++)
        {
            int v = (*all_edge[u])[i].first;
            int p1 = parent[u], p2 = parent[v];
            while (p1 != p2)
            { // link
                int h = p1 > p2 ? p1 : p2;
                int l = p2 >= p1 ? p1 : p2;
                int check = atomicCAS(&parent[h], h, l);
                if (check == h)
                {
                    break;
                }
                p1 = parent[parent[h]];
                p2 = parent[l];
            }
        }
    }
}

std::vector<int> WCC_GPU(GPU_adj<double> &input_graph)
{
    int N = input_graph.V; // number of vertices in the graph

    dim3 init_label_block((N + WCCG_THREAD_PER_BLOCK - 1) / WCCG_THREAD_PER_BLOCK, 1, 1); // the number of blocks used in the gpu
    dim3 init_label_thread(WCCG_THREAD_PER_BLOCK, 1, 1);                                  // the number of threads used in the gpu

    auto all_edge = input_graph.in_edge(); // graph stored in gpu adjacency list

    int *parent = nullptr;
    int *freq = nullptr;

    cudaMallocManaged((void **)&parent, N * sizeof(int));
    cudaMallocManaged((void **)&freq, N * sizeof(int));
    cudaMemset(freq, 0, N * sizeof(int));

    parent_init<<<init_label_block, init_label_thread>>>(parent, N);
    cudaDeviceSynchronize();

    int it = 0, ITERATION = 2; // number of iterations
    while (it < ITERATION)     // continue for a fixed number of iterations
    {
        sampling<<<init_label_block, init_label_thread>>>(all_edge, parent, N, it);
        cudaDeviceSynchronize();
        compress<<<init_label_block, init_label_thread>>>(parent, N);
        cudaDeviceSynchronize();
        it++;
    }
    get_freq<<<init_label_block, init_label_thread>>>(parent, freq, N);
    int *c = thrust::max_element(thrust::device, freq, freq + N);
    int most_f_element = *c;
    full_link<<<init_label_block, init_label_thread>>>(all_edge, parent, most_f_element, N, ITERATION);
    cudaDeviceSynchronize();
    compress<<<init_label_block, init_label_thread>>>(parent, N);
    cudaDeviceSynchronize();
    
    std::vector<int> result(N);
    cudaMemcpy(result.data(), parent, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(parent);
    cudaFree(freq);
    return result;
}

std::vector<std::pair<std::string, std::string>> Cuda_WCC_adj(graph_structure<double> &graph, GPU_adj<double> &adj_graph)
{
    std::vector<int> wccVecGPU = WCC_GPU(adj_graph);
    return graph.res_trans_id_id(wccVecGPU);
}

#endif