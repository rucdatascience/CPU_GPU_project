#pragma once

#include "cub/cub.cuh"
#include <stdio.h>
#include <stdlib.h>

#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
}

std::vector<std::pair<std::string, int>> Cuda_BFS_pre(graph_structure<double> &graph, CSR_graph<double> &csr_graph, std::string src_v, std::vector<int> &pre_v, int min_depth = 0, int max_depth = INT_MAX);

// Forward Traversal Kernel
__global__ void forwardKernel_pre(int *row_offsets, int *column_indices, int *depth, int *new_frontier, int *count, 
        int current_depth, int* frontier, int frontier_size, int *pre, int max_depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int node = frontier[tid];
    for (int i = row_offsets[node]; i < row_offsets[node + 1]; ++i) {
        int neighbor = column_indices[i];
        if (atomicCAS(&depth[neighbor], max_depth, current_depth + 1) == max_depth) {
            int pos = atomicAdd(count, 1);
            pre[neighbor] = node;
            new_frontier[pos] = neighbor;
        }
    }
}

// Reverse Traversal Kernel
__global__ void reverseKernel_pre(int *row_offsets, int *column_indices, int *depth, int *new_frontier, int *count, 
        int current_depth, int num_nodes, int *pre, int max_depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    if (depth[tid] == max_depth) {
        for (int i = row_offsets[tid]; i < row_offsets[tid + 1]; ++i) {
            int neighbor = column_indices[i];
            if (depth[neighbor] == current_depth) {
                depth[tid] = current_depth + 1;
                pre[tid] = neighbor;
                int pos = atomicAdd(count, 1);
                new_frontier[pos] = tid;
                break;
            }
        }
    }
}

__global__ void init_label_pre(int *label, int *pre_v, int node_size, int max_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < node_size) {
        label[idx] = max_depth;
        pre_v[idx] = -1;
    }
}

__host__ std::vector<int> bfs_pre(int *d_row_offsets_out, int *d_column_indices_out, int *d_row_offsets_in, int *d_column_indices_in,
        int num_nodes, int start_node, std::vector<int> &pre_v, int max_depth) {
    std::vector<int> result(num_nodes);
    
    int *d_depth, *d_frontier, *d_new_frontier, *d_count, *pre;
    cudaMalloc((void**)&d_depth, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_frontier, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_new_frontier, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMalloc((void**)&pre, num_nodes * sizeof(int));
    cudaDeviceSynchronize();
    
    // Initialize depth to -1
    // cudaMemcpy(d_depth, &init_val, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemset(d_depth, -2, num_nodes * sizeof(int));
    int blocks = (num_nodes + 255) / 256;
    init_label_pre<<<blocks, 256>>>(d_depth, pre, num_nodes, max_depth);
    cudaDeviceSynchronize();
    
    cudaMemcpy(d_frontier, &start_node, sizeof(int), cudaMemcpyHostToDevice);
    int frontier_size = 1;
    // cudaMemcpy(d_depth + start_node, &init_val, sizeof(int), cudaMemcpyHostToDevice); // depth[start_node] = 0
    cudaMemset(d_depth + start_node, 0, sizeof(int));
    cudaDeviceSynchronize();
    
    int current_depth = 0;

    while (frontier_size > 0) {
        // Heuristic threshold: switch direction when frontier size is more than 1/100 of total nodes
        bool use_forward = (frontier_size < num_nodes / 100);
        
        if (use_forward) {
            int blocks = (frontier_size + 255) / 256;
            cudaMemset(d_count, 0, sizeof(int));
            forwardKernel_pre<<<blocks, 256>>>(d_row_offsets_out, d_column_indices_out, d_depth, d_new_frontier, d_count, current_depth, d_frontier, frontier_size, pre, max_depth);
            cudaDeviceSynchronize();
            cudaMemcpy(&frontier_size, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        } else {
            int blocks = (num_nodes + 255) / 256;
            cudaMemset(d_count, 0, sizeof(int));
            reverseKernel_pre<<<blocks, 256>>>(d_row_offsets_in, d_column_indices_in, d_depth, d_new_frontier, d_count, current_depth, num_nodes, pre, max_depth);
            cudaDeviceSynchronize();
            cudaMemcpy(&frontier_size, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        }

        // Swap frontiers
        int *temp = d_frontier;
        d_frontier = d_new_frontier;
        d_new_frontier = temp;
        // if (current_depth % 10000 == 0) printf("current_depth: %d\n", current_depth);
        // if (current_depth >= 100000) break;
        current_depth++;
    }
    cudaMemcpy(result.data(), d_depth, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pre_v.data(), pre, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(d_depth);
    cudaFree(d_frontier);
    cudaFree(d_new_frontier);
    cudaFree(d_count);
    cudaFree(pre);
    return result;
}

std::vector<std::pair<std::string, int>> Cuda_BFS_pre(graph_structure<double> &graph, CSR_graph<double> &csr_graph, std::string src_v, std::vector<int> &pre_v, int min_depth, int max_depth){
    int V = csr_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = csr_graph.OUTs_Edges.size();
    pre_v.resize(graph.V);
    int src_v_id = graph.vertex_str_to_id[src_v];
    std::vector<int> bfsVecGPU = bfs_pre(csr_graph.out_pointer, csr_graph.out_edge, csr_graph.in_pointer, csr_graph.in_edge, V, src_v_id, pre_v, max_depth);
    
    for (int i = 0; i < graph.V; i++) {
		int dep = bfsVecGPU[i];
		int pre = pre_v[i];
        if (pre == -1) continue; // no path
        int now = i;
		while (pre != -1) {
			bool ff = false;
			for (auto edge : graph.OUTs[pre]) {
				if (edge.first == now) {
                    now = pre;
                    pre = pre_v[pre];
					ff = true;
					break;
				}
			}
			if (!ff) {
                printf("Not found!\n");
				break;
			}
		}
	}

    return graph.res_trans_id_val(bfsVecGPU); // return the results in string type
}