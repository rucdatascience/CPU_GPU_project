#ifndef WS_SSSP_H_ADJ
#define WS_SSSP_H_ADJ

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GPU_adj_list/GPU_adj.hpp>

__device__ __forceinline__ double atomicMinDouble (double * addr, double value);

__global__ void Relax(int* offsets, int* edges, double* weights, double* dis, int* queue, int* queue_size, int* visited);
__global__ void CompactQueue(int V, int* next_queue, int* next_queue_size, int* visited);
void gpu_shortest_paths(GPU_adj<double>& input_graph, int source, std::vector<double>& distance, double max_dis = 10000000000);

std::vector<std::pair<std::string, double>> Cuda_SSSP_adj(graph_structure<double>& graph, GPU_adj<double>& gpu_adj, std::string src_v, double max_dis = 10000000000);

// this function is used to get the minimum value of double type atomically
__device__ __forceinline__ double atomicMinDouble (double * addr, double value) {
    double old;
    old = __longlong_as_double(atomicMin((long long *)addr, __double_as_longlong(value)));
    return old;
}

__global__ void Relax(cuda_vector<int>** out_edge, cuda_vector<double>** out_edge_weight, double* dis, int* queue, int* queue_size, int* visited) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *queue_size) {
        int v = queue[idx];

        // for all adjacent vertices
        int out_edge_size = out_edge[v]->size();
        cuda_vector<int>& ed = *out_edge[v];
        cuda_vector<double>& ed_weight = *out_edge_weight[v];
        for (int i = 0; i < out_edge_size; i++) {
            int new_v = ed[i];
            double weight = ed_weight[i];

            double new_w = dis[v] + weight;

            // try doing relaxation
            double old = atomicMinDouble(&dis[new_v], new_w);

            if (old <= new_w)
				continue;

            // if the distance is updated, set the vertex as visited
            atomicExch(&visited[new_v], 1);
        }
    }
}

__global__ void CompactQueue(int V, int* next_queue, int* next_queue_size, int* visited) {
    // this function is used to ensure that each necessary vertex is only pushed into the queue once
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V && visited[idx]) {
        int pos = atomicAdd(next_queue_size, 1);
        next_queue[pos] = idx;
        // reset the visited flag
        visited[idx] = 0;
    }
}

void gpu_shortest_paths(GPU_adj<double>& input_graph, int source, std::vector<double>& distance, double max_dis) {
    int V = input_graph.V;

    double* dis;
    auto out_edge = input_graph.out_edge();
    auto out_edge_weight = input_graph.out_edge_weight();
    int* visited;

    int* queue, * next_queue;
    int* queue_size, * next_queue_size;

    // allocate memory on GPU
    cudaMallocManaged((void**)&dis, V * sizeof(double));
    cudaMallocManaged((void**)&visited, V * sizeof(int));
    cudaMallocManaged((void**)&queue, V * sizeof(int));
    cudaMallocManaged((void**)&next_queue, V * sizeof(int));
    cudaMallocManaged((void**)&queue_size, sizeof(int));
    cudaMallocManaged((void**)&next_queue_size, sizeof(int));

    // synchronize the device
    cudaDeviceSynchronize();
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    for (int i = 0; i < V; i++) {
        // initialize the distance array and visited array
		dis[i] = max_dis;
		visited[i] = 0;
	}
    dis[source] = 0;


    *queue_size = 1;
    queue[0] = source;
    *next_queue_size = 0;

    int threadsPerBlock = 1024;
    int numBlocks = 0;

    while (*queue_size > 0) {
		numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
        // launch the kernel function to relax the edges
		Relax <<< numBlocks, threadsPerBlock >>> (out_edge, out_edge_weight, dis, queue, queue_size, visited);
		cudaDeviceSynchronize();

        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Relax kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

		numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;
        // do the compact operation
		CompactQueue <<< numBlocks, threadsPerBlock >>> (V, next_queue, next_queue_size, visited);
		cudaDeviceSynchronize();

        cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "CompactQueue kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
			return;
		}

        // swap the queue and next_queue
        std::swap(queue, next_queue);

		*queue_size = *next_queue_size;
        *next_queue_size = 0;
	}
    cudaMemcpy(distance.data(), dis, V * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free the memory
    cudaFree(dis);
    cudaFree(visited);
    cudaFree(queue);
    cudaFree(next_queue);
    cudaFree(queue_size);
    cudaFree(next_queue_size);
    return;
}

std::vector<std::pair<std::string, double>> Cuda_SSSP_adj(graph_structure<double>& graph, GPU_adj<double>& gpu_adj, std::string src_v, double max_dis) {
    int src_v_id = graph.vertex_str_to_id[src_v];
    std::vector<double> gpuSSSPvec(graph.V, 0);
    gpu_shortest_paths(gpu_adj, src_v_id, gpuSSSPvec, max_dis);

    // transfer the vertex id to vertex name
    return graph.res_trans_id_val(gpuSSSPvec);
}

#endif