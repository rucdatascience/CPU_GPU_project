#pragma once

#include "cub/cub.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <GPU_gpma/GPU_gpma.hpp>
#include <GPU_csr/GPU_csr.hpp>

#define FULL_MASK 0xffffffff

std::vector<std::pair<std::string, int>> Cuda_BFS(graph_structure<double> &graph, CSR_graph<double> &csr_graph, std::string src_v, int min_depth = 0, int max_depth = INT_MAX);

template<int THREADS_NUM>
__global__ void csr_bfs_gather_kernel(int *node_queue, int *node_queue_offset,
        int *edge_queue, int *edge_queue_offset,
        int *keys, double *values, int *row_offsets, int *dep, int max_depth) {

    typedef cub::BlockScan<int, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;

    volatile __shared__ int comm[THREADS_NUM / 32][3];
    volatile __shared__ int comm2[THREADS_NUM];
    volatile __shared__ int output_cta_offset;
    volatile __shared__ int output_warp_offset[THREADS_NUM / 32];

    typedef cub::WarpScan<int> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[THREADS_NUM / 32];

    int thread_id = threadIdx.x;
    int lane_id = thread_id % 32;
    int warp_id = thread_id / 32;

    int cta_offset = blockDim.x * blockIdx.x;
    while (cta_offset < node_queue_offset[0]) {
        int node, row_begin, row_end;
        if (cta_offset + thread_id < node_queue_offset[0]) {
            node = node_queue[cta_offset + thread_id];
            row_begin = row_offsets[node];
            row_end = row_offsets[node + 1];
        } else
            row_begin = row_end = 0;

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
                row_begin = row_end;
            }
            __syncthreads();

            int gather = comm[0][1] + thread_id;
            int gather_end = comm[0][2];
            int neighbour;
            int thread_data_in;
            int thread_data_out;
            int block_aggregate;
            while (__syncthreads_or(gather < gather_end)) {
                if (gather < gather_end) {
                    neighbour = keys[gather];
                    thread_data_in = 1;
                } else
                    thread_data_in = 0;

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
                row_begin = row_end;
            }

            int gather = comm[warp_id][1] + lane_id;
            int gather_end = comm[warp_id][2];
            int neighbour;
            int thread_data_in;
            int thread_data_out;
            int warp_aggregate;
            while (__any_sync(FULL_MASK, gather < gather_end)) {
                if (gather < gather_end) {
                    neighbour = keys[gather];
                    thread_data_in = 1;
                } else
                    thread_data_in = 0;

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
        int thread_data = row_end - row_begin;
        int rsv_rank;
        int total;
        int remain;
        __syncthreads();
        BlockScan(block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        int cta_progress = 0;
        while (cta_progress < total) {
            remain = total - cta_progress;

            // share batch of gather offsets
            while ((rsv_rank < cta_progress + THREADS_NUM) && (row_begin < row_end)) {
                comm2[rsv_rank - cta_progress] = row_begin;
                rsv_rank++;
                row_begin++;
            }
            __syncthreads();
            int neighbour;
            // gather batch of adjlist
            if (thread_id < min(remain, THREADS_NUM)) {
                neighbour = keys[comm2[thread_id]];
                thread_data = 1;
            } else
                thread_data = 0;
            __syncthreads();

            int scatter;
            int block_aggregate;

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

template<int THREADS_NUM>
__global__ void csr_bfs_contract_kernel(int *edge_queue, int *edge_queue_offset, int *node_queue, int *node_queue_offset, 
        int level, int *label, int max_depth, int *bitmap) {

    typedef cub::BlockScan<int, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    volatile __shared__ int output_cta_offset;

    volatile __shared__ int warp_cache[THREADS_NUM / 32][128];
    const int HASH_KEY1 = 1097;
    const int HASH_KEY2 = 1103;
    volatile __shared__ int cta1_cache[HASH_KEY1];
    volatile __shared__ int cta2_cache[HASH_KEY2];

    // init cta-level cache
    for (int i = threadIdx.x; i < HASH_KEY1; i += blockDim.x)
        cta1_cache[i] = SIZE_NONE;
    for (int i = threadIdx.x; i < HASH_KEY2; i += blockDim.x)
        cta2_cache[i] = SIZE_NONE;
    __syncthreads();

    int thread_id = threadIdx.x;
    int warp_id = thread_id / 32;
    int cta_offset = blockDim.x * blockIdx.x;

    while (cta_offset < edge_queue_offset[0]) {
        int neighbour;
        int valid = 0;

        do {
            if (cta_offset + thread_id >= edge_queue_offset[0]) break;
            neighbour = edge_queue[cta_offset + thread_id];

            // warp cull
            int hash = neighbour & 127;
            warp_cache[warp_id][hash] = neighbour;
            int retrieved = warp_cache[warp_id][hash];
            if (retrieved == neighbour) {
                warp_cache[warp_id][hash] = thread_id;
                if (warp_cache[warp_id][hash] != thread_id)
                    break;
            }

            // history cull
            if (cta1_cache[neighbour % HASH_KEY1] == neighbour) break;
            if (cta2_cache[neighbour % HASH_KEY2] == neighbour) break;
            cta1_cache[neighbour % HASH_KEY1] = neighbour;
            cta2_cache[neighbour % HASH_KEY2] = neighbour;

            // bitmap check
            int bit_loc = 1 << (neighbour % 32);
            int bit_chunk = bitmap[neighbour / 32];
            if (bit_chunk & bit_loc) break;
            bitmap[neighbour / 32] = bit_chunk + bit_loc;

            int ret = atomicCAS(label + neighbour, max_depth, level);
            valid = (ret != max_depth) ? 0 : 1;
        } while (false);
        __syncthreads();

        int scatter;
        int total;
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

__global__ void init_label(int *label, int node_size, int max_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < node_size) {
        label[idx] = max_depth;
    }
}

__host__ std::vector<int> csr_bfs(int *keys, double *values, int *row_offsets, int node_size, int edge_size, 
        int start_node, int max_depth) {
    std::vector<int> result(node_size, std::numeric_limits<double>::max());
    const int THREADS_NUM = 256;
    int BLOCKS_NUM = (1 + THREADS_NUM - 1) / THREADS_NUM;
    int *results;
    cudaMallocManaged(&results, sizeof(int) * node_size);
    cudaDeviceSynchronize();

    cudaMemcpy(results, result.data(), node_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    int *bitmap;
    cudaMalloc(&bitmap, sizeof(int) * ((node_size - 1) / 32 + 1));
    cudaDeviceSynchronize();
    cudaMemset(bitmap, 0, sizeof(int) * ((node_size - 1) / 32 + 1));
    int *node_queue;
    cudaMalloc(&node_queue, sizeof(int) * node_size);
    int *node_queue_offset;
    cudaMalloc(&node_queue_offset, sizeof(int));
    int *edge_queue;
    cudaMalloc(&edge_queue, sizeof(int) * edge_size);
    int *edge_queue_offset;
    cudaMalloc(&edge_queue_offset, sizeof(int));
    cudaDeviceSynchronize();
    
    // init
    int host_num[1];
    host_num[0] = start_node;
    cudaMemcpy(node_queue, host_num, sizeof(int), cudaMemcpyHostToDevice);
    host_num[0] = 1 << (start_node % 32);
    cudaMemcpy(&bitmap[start_node / 32], host_num, sizeof(int), cudaMemcpyHostToDevice);
    host_num[0] = 1;
    cudaMemcpy(node_queue_offset, host_num, sizeof(int), cudaMemcpyHostToDevice);
    host_num[0] = 0;
    cudaMemcpy(&results[start_node], host_num, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    int level = 1;
    float time_relax = 0.0, time_compact = 0.0, milliseconds;
    while (true) {
        // gather
        int BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, host_num[0]);
        host_num[0] = 0;
        cudaMemcpy(edge_queue_offset, host_num, sizeof(int), cudaMemcpyHostToDevice);
        csr_bfs_gather_kernel<THREADS_NUM> <<<BLOCKS_NUM, THREADS_NUM>>>(node_queue, node_queue_offset,
                edge_queue, edge_queue_offset, keys, values, row_offsets, results, max_depth);
        cudaDeviceSynchronize();
    
        // contract
        level++;
        cudaMemcpy(node_queue_offset, host_num, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(host_num, edge_queue_offset, sizeof(int), cudaMemcpyDeviceToHost);
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, host_num[0]);

        csr_bfs_contract_kernel<THREADS_NUM> <<<BLOCKS_NUM, THREADS_NUM>>>(edge_queue, edge_queue_offset,
                node_queue, node_queue_offset, level - 1, results, max_depth, bitmap);
        cudaMemcpy(host_num, node_queue_offset, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    
        if (0 == host_num[0]) break;
    }
    cudaMemcpy(result.data(), results, node_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(results);
    cudaFree(bitmap);
    cudaFree(node_queue);
    cudaFree(node_queue_offset);
    cudaFree(edge_queue);
    cudaFree(edge_queue_offset);
    return result;
}

std::vector<std::pair<std::string, int>> Cuda_BFS(graph_structure<double> &graph, CSR_graph<double> &csr_graph, std::string src_v, int min_depth, int max_depth){
    int V = graph.V;
    int E = csr_graph.OUTs_Edges.size();
    int src_v_id = graph.vertex_str_to_id[src_v];
    std::vector<int> bfsVecGPU = csr_bfs(csr_graph.out_edge, csr_graph.out_edge_weight, csr_graph.out_pointer, V, E, src_v_id, max_depth);
    return graph.res_trans_id_val(bfsVecGPU); // return the results in string type
}