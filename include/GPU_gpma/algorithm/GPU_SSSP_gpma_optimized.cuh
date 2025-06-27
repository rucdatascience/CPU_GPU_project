#pragma once
#include "cub/cub.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GPU_gpma/GPU_gpma.hpp>

#define FULL_MASK 0xffffffff

// 启动SSSP的主机函数
std::vector<double> gpma_sssp(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offsets,
                             SIZE_TYPE node_size, SIZE_TYPE edge_size, SIZE_TYPE start_node, double INF = 10000000000);
std::vector<std::pair<std::string, double>> Cuda_SSSP_optimized(graph_structure<double> &graph, GPMA& gpma, std::string src_v, double max_dis = 10000000000);

__device__ __forceinline__ double _atomicMinDouble(double* address, double value) {
    double old;
    old = __longlong_as_double(atomicMin((long long *)address, __double_as_longlong(value)));
    return old;
}

template<SIZE_TYPE THREADS_NUM>
__global__ void gpma_sssp_gather_kernel(SIZE_TYPE *node_queue, SIZE_TYPE *node_queue_offset,
                                        KEY_TYPE *edge_queue, SIZE_TYPE *edge_queue_offset,
                                        KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offsets, double *distances, int *bitmap) {
    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;

    volatile __shared__ SIZE_TYPE comm[THREADS_NUM / 32][3];
    volatile __shared__ SIZE_TYPE comm2[THREADS_NUM];
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
                row_begin = row_end;
            }
            __syncthreads();

            SIZE_TYPE gather = comm[0][1] + thread_id;
            SIZE_TYPE gather_end = comm[0][2];
            SIZE_TYPE neighbour, u, v;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE block_aggregate;
            while (__syncthreads_or(gather < gather_end)) {
                if (gather < gather_end) {
                    KEY_TYPE cur_key = keys[gather];
                    VALUE_TYPE cur_value = values[gather];
                    neighbour = gather;
                    
                    u = (cur_key >> 32);
                    v = (SIZE_TYPE) (cur_key & COL_IDX_NONE);
                    if (v == COL_IDX_NONE || cur_value == VALUE_NONE || distances[u] + cur_value >= distances[v]) {
                        thread_data_in = 0;
                    } else {
                        thread_data_in = 1;
                    }
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

            SIZE_TYPE gather = comm[warp_id][1] + lane_id;
            SIZE_TYPE gather_end = comm[warp_id][2];
            SIZE_TYPE neighbour, u, v;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE warp_aggregate;
            while (__any_sync(FULL_MASK, gather < gather_end)) {
                if (gather < gather_end) {
                    KEY_TYPE cur_key = keys[gather];
                    VALUE_TYPE cur_value = values[gather];
                    neighbour = gather;
                    u = (cur_key >> 32);
                    v = (SIZE_TYPE) (cur_key & COL_IDX_NONE);
                    if (v == COL_IDX_NONE || cur_value == VALUE_NONE || distances[u] + cur_value >= distances[v]) {
                        thread_data_in = 0;
                    } else {
                        thread_data_in = 1;
                    }
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
                comm2[rsv_rank - cta_progress] = row_begin;
                rsv_rank ++;
                row_begin ++;
            }
            __syncthreads();
            SIZE_TYPE neighbour, u, v;
            // gather batch of adjlist
            if (thread_id < min(remain, THREADS_NUM)) {
                KEY_TYPE cur_key = keys[comm2[thread_id]];
                VALUE_TYPE cur_value = values[comm2[thread_id]];
                neighbour = comm2[thread_id];
                
                u = (cur_key >> 32);
                v = (SIZE_TYPE) (cur_key & COL_IDX_NONE);
                if (v == COL_IDX_NONE || cur_value == VALUE_NONE || distances[u] + cur_value >= distances[v]) {
                    thread_data = 0;
                } else {
                    thread_data = 1;
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
__global__ void gpma_sssp_relax_kernel(KEY_TYPE *edge_queue, SIZE_TYPE *edge_queue_offset,
                                      SIZE_TYPE *node_queue, SIZE_TYPE *node_queue_offset,
                                      double *distances, KEY_TYPE *keys, VALUE_TYPE *values, int *bitmap) {
    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    volatile __shared__ SIZE_TYPE output_cta_offset;

    SIZE_TYPE thread_id = threadIdx.x;
    SIZE_TYPE cta_offset = blockDim.x * blockIdx.x;
    
    while (cta_offset < edge_queue_offset[0]) {
        SIZE_TYPE neighbour;
        SIZE_TYPE edge_idx = cta_offset + thread_id;
        double old_dist, new_dist;
        SIZE_TYPE u, v;
        double weight;
        bool valid = false;
        
        if (edge_idx < edge_queue_offset[0]) {
            edge_idx = edge_queue[cta_offset + thread_id];
            u = keys[edge_idx] >> 32;
            v = (SIZE_TYPE) (keys[edge_idx] & COL_IDX_NONE);
            neighbour = v;
            
            weight = values[edge_idx];
            old_dist = distances[u];
            new_dist = old_dist + weight;
            
            // double expected = _atomicMinDouble(&distances[v], new_dist);
            _atomicMinDouble(&distances[v], new_dist);
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

__global__ void init_array(double* array, SIZE_TYPE size, double value) {
    SIZE_TYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = value;
    }
}

__host__ std::vector<double> gpma_sssp(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offsets,
                                     SIZE_TYPE node_size, SIZE_TYPE edge_size, SIZE_TYPE start_node, double INF) {
    const SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM = (node_size + THREADS_NUM - 1) / THREADS_NUM;

    double *distances;
    cudaMalloc(&distances, sizeof(double) * node_size);
    cudaDeviceSynchronize();
    init_array<<<BLOCKS_NUM, THREADS_NUM>>>(distances, node_size, INF);

    SIZE_TYPE *node_queue, *node_queue_offset;
    SIZE_TYPE *edge_queue_offset;
    KEY_TYPE *edge_queue;
    int *bitmap;
    cudaMalloc(&node_queue, sizeof(SIZE_TYPE) * node_size);
    cudaMalloc(&node_queue_offset, sizeof(SIZE_TYPE));
    cudaMalloc(&edge_queue, sizeof(KEY_TYPE) * edge_size);
    cudaMalloc(&edge_queue_offset, sizeof(SIZE_TYPE));
    cudaMalloc(&bitmap, sizeof(int) * node_size);
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
        gpma_sssp_gather_kernel<THREADS_NUM><<<BLOCKS_NUM, THREADS_NUM>>>(node_queue, node_queue_offset, edge_queue, edge_queue_offset, keys, values, row_offsets, distances, bitmap);
        cudaDeviceSynchronize();

        cudaMemcpy(node_queue_offset, &zero, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        gpma_sssp_relax_kernel<THREADS_NUM><<<BLOCKS_NUM, THREADS_NUM>>>(edge_queue, edge_queue_offset, node_queue, node_queue_offset, distances, keys, values, bitmap);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&zero, node_queue_offset, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        
        if (zero == 0) break;
    }

    std::vector<double> result(node_size);
    cudaMemcpy(result.data(), distances, node_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(node_queue);
    cudaFree(node_queue_offset);
    cudaFree(edge_queue);
    cudaFree(edge_queue_offset);
    cudaFree(bitmap);
    cudaFree(distances);
    return result;
}

std::vector<std::pair<std::string, double>> Cuda_SSSP_optimized(graph_structure<double> &graph, GPMA& gpma, std::string src_v, double max_dis) {
    int V = gpma.get_V();
    int E = gpma.get_size();
    int src_v_id = graph.vertex_str_to_id[src_v];
    std::vector<double> ssspVecGPU = gpma_sssp(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(gpma.row_offset), V, E, src_v_id, max_dis);
    return graph.res_trans_id_val(ssspVecGPU);
}