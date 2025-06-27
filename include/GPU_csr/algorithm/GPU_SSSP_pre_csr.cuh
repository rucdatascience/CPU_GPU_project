#pragma once

#include "cub/cub.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <GPU_gpma/GPU_gpma.hpp>
#include <GPU_csr/GPU_csr.hpp>
#include <GPU_csr/algorithm/GPU_SSSP_csr.cuh>

#define FULL_MASK 0xffffffff

std::vector<double> csr_sssp_pre(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offsets,
                             SIZE_TYPE node_size, SIZE_TYPE edge_size, SIZE_TYPE start_node, double INF = 10000000000);
std::vector<std::pair<std::string, double>> Cuda_SSSP_pre(graph_structure<double> &graph, GPMA& gpma, std::string src_v, double max_dis = 10000000000);

template<SIZE_TYPE THREADS_NUM>
__global__ void csr_sssp_pre_gather_kernel(SIZE_TYPE *node_queue, SIZE_TYPE *node_queue_offset,
                                        KEY_TYPE *edge_queue, SIZE_TYPE *edge_queue_offset,
                                        int *keys, double *values, int *row_offsets, double *distances, int *bitmap) {
    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;

    volatile __shared__ SIZE_TYPE comm[THREADS_NUM / 32][4];
    volatile __shared__ SIZE_TYPE comm2[THREADS_NUM][2];
    volatile __shared__ SIZE_TYPE output_cta_offset;
    volatile __shared__ SIZE_TYPE output_warp_offset[THREADS_NUM / 32];

    typedef cub::WarpScan<SIZE_TYPE> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[THREADS_NUM / 32];

    SIZE_TYPE thread_id = threadIdx.x;
    SIZE_TYPE lane_id = thread_id % 32;
    SIZE_TYPE warp_id = thread_id / 32;
    
    SIZE_TYPE cta_offset = blockDim.x * blockIdx.x;
    while (cta_offset < node_queue_offset[0]) {
        SIZE_TYPE node, row_begin, row_end;
        if (cta_offset + thread_id < node_queue_offset[0]) {
            node = node_queue[cta_offset + thread_id];
            row_begin = row_offsets[node];
            row_end = row_offsets[node + 1];
            bitmap[node] = 0;
        } else {
            row_begin = row_end = 0;
        }
        // CTA-based coarse-grained gather
        while (__syncthreads_or(row_end - row_begin >= THREADS_NUM)) {
            // vie for control of block
            if (row_end - row_begin >= THREADS_NUM)
                comm[0][0] = thread_id;
            __syncthreads();

            // winner describes adjlist
            if (comm[0][0] == thread_id) {
                comm[0][1] = row_begin;
                comm[0][2] = row_end;
                comm[0][3] = node;
                row_begin = row_end;
            }
            __syncthreads();

            SIZE_TYPE gather = comm[0][1] + thread_id;
            SIZE_TYPE gather_end = comm[0][2];
            KEY_TYPE neighbour;
            SIZE_TYPE u = comm[0][3];
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE block_aggregate;
            while (__syncthreads_or(gather < gather_end)) {
                if (gather < gather_end && distances[u] + values[gather] < distances[keys[gather]]) {
                    thread_data_in = 1;
                    neighbour = ((KEY_TYPE)u << 32) | gather;
                } else {
                    thread_data_in = 0;
                }
                __syncthreads();
                
                BlockScan(block_temp_storage).ExclusiveSum(thread_data_in, thread_data_out, block_aggregate);
                __syncthreads();
                
                if (0 == thread_id) {
                    output_cta_offset = atomicAdd(edge_queue_offset, block_aggregate);
                }
                __syncthreads();
                
                if (thread_data_in)
                    edge_queue[output_cta_offset + thread_data_out] = neighbour;
                gather += THREADS_NUM;
            }
        }

        // warp-based coarse-grained gather
        while (__any_sync(FULL_MASK, row_end - row_begin >= 32)) {
            // vie for control of warp
            if (row_end - row_begin >= 32)
                comm[warp_id][0] = lane_id;

            // winner describes adjlist
            if (comm[warp_id][0] == lane_id) {
                comm[warp_id][1] = row_begin;
                comm[warp_id][2] = row_end;
                comm[warp_id][3] = node;
                row_begin = row_end;
            }

            SIZE_TYPE gather = comm[warp_id][1] + lane_id;
            SIZE_TYPE gather_end = comm[warp_id][2];
            KEY_TYPE neighbour;
            SIZE_TYPE u = comm[warp_id][3];
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE warp_aggregate;
            while (__any_sync(FULL_MASK, gather < gather_end)) {
                if (gather < gather_end && distances[u] + values[gather] < distances[keys[gather]]) {
                    thread_data_in = 1;
                    neighbour = ((KEY_TYPE)u << 32) | gather; // keys[gather];
                } else {
                    thread_data_in = 0;
                }

                WarpScan(temp_storage[warp_id]).ExclusiveSum(thread_data_in, thread_data_out, warp_aggregate);

                if (0 == lane_id) {
                    output_warp_offset[warp_id] = atomicAdd(edge_queue_offset, warp_aggregate);
                }

                if (thread_data_in)
                    edge_queue[output_warp_offset[warp_id] + thread_data_out] = neighbour;
                gather += 32;
            }
        }

        // scan-based fine-grained gather
        SIZE_TYPE thread_data = row_end - row_begin;
        SIZE_TYPE rsv_rank;
        SIZE_TYPE total;
        SIZE_TYPE remain;
        __syncthreads();
        BlockScan(block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        SIZE_TYPE cta_progress = 0;
        while (cta_progress < total) {
            remain = total - cta_progress;

            // share batch of gather offsets
            while ((rsv_rank < cta_progress + THREADS_NUM) && (row_begin < row_end)) {
                comm2[rsv_rank - cta_progress][0] = row_begin;
                comm2[rsv_rank - cta_progress][1] = node;
                rsv_rank ++;
                row_begin ++;
            }
            __syncthreads();
            KEY_TYPE neighbour;
            SIZE_TYPE u = comm2[thread_id][1], edge_id = comm2[thread_id][0];
            // gather batch of adjlist
            if (thread_id < min(remain, THREADS_NUM) && distances[u] + values[edge_id] < distances[keys[edge_id]]) {
                if (distances[u] + values[edge_id] >= distances[keys[edge_id]]) {
                    thread_data = 0;
                } else {
                    thread_data = 1;
                    neighbour = ((KEY_TYPE)u << 32) | edge_id; // keys[comm2[thread_id]];
                }
            } else
                thread_data = 0;
            __syncthreads();

            SIZE_TYPE scatter;
            SIZE_TYPE block_aggregate;

            BlockScan(block_temp_storage).ExclusiveSum(thread_data, scatter, block_aggregate);
            __syncthreads();

            if (0 == thread_id) {
                output_cta_offset = atomicAdd(edge_queue_offset, block_aggregate);
            }
            __syncthreads();

            if (thread_data)
                edge_queue[output_cta_offset + scatter] = neighbour;
            cta_progress += THREADS_NUM;
            __syncthreads();
        }

        cta_offset += blockDim.x * gridDim.x;
    }
}

template<SIZE_TYPE THREADS_NUM>
__global__ void csr_sssp_pre_relax_kernel(KEY_TYPE *edge_queue, SIZE_TYPE *edge_queue_offset, SIZE_TYPE *node_queue, 
        SIZE_TYPE *node_queue_offset, double *distances, int *keys, double *values, int *pre, int *mutex, int *bitmap) {
    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    volatile __shared__ SIZE_TYPE output_cta_offset;

    SIZE_TYPE thread_id = threadIdx.x;
    SIZE_TYPE cta_offset = blockDim.x * blockIdx.x;
    
    while (cta_offset < edge_queue_offset[0]) {
        SIZE_TYPE neighbour;
        SIZE_TYPE valid = 0;
        KEY_TYPE edge_idx = cta_offset + thread_id;
        int u, v;
        if (edge_idx < edge_queue_offset[0]) {
            edge_idx = edge_queue[cta_offset + thread_id];
            
            u = (int)(edge_idx >> 32);
            edge_idx &= COL_IDX_NONE;
            neighbour = v = keys[edge_idx];
            
            double new_dis = distances[u] + values[edge_idx];
            
            while (atomicCAS(&mutex[v], 0, 1) != 0);
            if (new_dis < distances[v]) {
                distances[v] = new_dis;
                // update the previous vertex
                pre[v] = u;
                // share the updated distance with other threads in different blocks
                __threadfence();
            }
            atomicExch(&mutex[v], 0);
            
            if (atomicExch(&bitmap[v], 1) == 0) {
                valid = 1;
            }
        }
        __syncthreads();

        SIZE_TYPE scatter;
        SIZE_TYPE total;
        BlockScan(temp_storage).ExclusiveSum(valid, scatter, total);
        __syncthreads();

        if (0 == thread_id) {
            output_cta_offset = atomicAdd(node_queue_offset, total);
        }
        __syncthreads();

        if (valid)
            node_queue[output_cta_offset + scatter] = neighbour;

        cta_offset += blockDim.x * gridDim.x;
    }
}

__global__ void init_array_sssp_pre(double* array, int* pre_v, SIZE_TYPE size, double value) {
    SIZE_TYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = value;
        pre_v[idx] = -1;
    }
}

__host__ std::vector<double> csr_sssp_pre(int *keys, double *values, int *row_offsets,
        SIZE_TYPE node_size, SIZE_TYPE edge_size, SIZE_TYPE start_node, std::vector<int>& pre_v, double INF) {
    const SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM = (node_size + THREADS_NUM - 1) / THREADS_NUM;

    double *distances;
    int *pre;
    cudaMalloc(&distances, sizeof(double) * node_size);
    cudaMalloc(&pre, sizeof(int) * node_size);
    cudaDeviceSynchronize();
    init_array_sssp_pre<<<BLOCKS_NUM, THREADS_NUM>>>(distances, pre, node_size, INF);

    SIZE_TYPE *node_queue, *node_queue_offset;
    SIZE_TYPE *edge_queue_offset;
    KEY_TYPE *edge_queue;
    int *mutex;
    int *bitmap;
    cudaMalloc(&node_queue, sizeof(SIZE_TYPE) * node_size);
    cudaMalloc(&node_queue_offset, sizeof(SIZE_TYPE));
    cudaMalloc(&edge_queue, sizeof(KEY_TYPE) * edge_size);
    cudaMalloc(&edge_queue_offset, sizeof(SIZE_TYPE));
    cudaMalloc(&bitmap, sizeof(int) * node_size);
    cudaMalloc(&mutex, sizeof(int) * node_size);
    cudaDeviceSynchronize();
    cudaMemset(bitmap, 0, sizeof(int) * node_size);

    // init
    SIZE_TYPE host_num[1];
    host_num[0] = start_node;
    cudaMemcpy(node_queue, host_num, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    host_num[0] = 1;
    cudaMemcpy(node_queue_offset, host_num, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    double host_num_2[1];
    host_num_2[0] = 0;
    cudaMemcpy(&distances[start_node], host_num_2, sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    int iter = 0;
    while (true) {
        iter ++;

        SIZE_TYPE zero = 0;
        cudaMemcpy(edge_queue_offset, &zero, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        csr_sssp_pre_gather_kernel<THREADS_NUM><<<BLOCKS_NUM, THREADS_NUM>>>(node_queue, node_queue_offset, 
            edge_queue, edge_queue_offset, keys, values, row_offsets, distances, bitmap);
        cudaDeviceSynchronize();
        
        cudaMemcpy(node_queue_offset, &zero, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        csr_sssp_pre_relax_kernel<THREADS_NUM><<<BLOCKS_NUM, THREADS_NUM>>>(edge_queue, edge_queue_offset, 
            node_queue, node_queue_offset, distances, keys, values, pre, mutex, bitmap);
        cudaDeviceSynchronize();

        cudaMemcpy(&zero, node_queue_offset, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (zero == 0) break;
    }

    std::vector<double> result(node_size);
    cudaMemcpy(result.data(), distances, node_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(pre_v.data(), pre, node_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(node_queue);
    cudaFree(node_queue_offset);
    cudaFree(edge_queue);
    cudaFree(edge_queue_offset);
    cudaFree(bitmap);
    cudaFree(distances);
    cudaFree(pre);
    cudaFree(mutex);
    return result;
}

std::vector<std::pair<std::string, double>> Cuda_SSSP_pre(graph_structure<double> &graph, CSR_graph<double> &csr_graph, std::string src_v, std::vector<int>& pre_v, double max_dis) {
    int V = csr_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = csr_graph.OUTs_Edges.size();
    int src_v_id = graph.vertex_str_to_id[src_v];
    pre_v.resize(graph.V);
    std::vector<double> ssspVecGPU = csr_sssp_pre(csr_graph.out_edge, csr_graph.out_edge_weight, csr_graph.out_pointer, V, E, src_v_id, pre_v, max_dis);
    
    // check the correctness of the previous vertex
    for (int i = 0; i < graph.V; i++) {
		double dis = ssspVecGPU[i];
		int pre = pre_v[i];
        if (pre == -1) continue; // no path
        int now = i;
		double sum = 0;
		while (pre != -1) {
			bool ff = false;
			for (auto edge : graph.OUTs[pre]) {
				if (edge.first == now) {
					sum += edge.second;
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
		if (fabs(sum - dis) > 1e-4) {
            printf("Error: pre_v is wrong!\nvertex: %d, sum: %.5lf, dis: %.5lf\n", i, sum, dis);
		}
	}

    return graph.res_trans_id_val(ssspVecGPU); // 假设存在对应的转换函数
}