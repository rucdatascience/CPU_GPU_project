#pragma once
#include "cub/cub.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GPU_gpma/GPU_gpma.hpp>

template<SIZE_TYPE THREADS_NUM>
__global__ void gpma_cdlp_gather_kernel(KEY_TYPE *edge_queue, SIZE_TYPE *edge_queue_offset,
        KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offsets,  
        SIZE_TYPE *label, SIZE_TYPE *node_queue_offset) {

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
            node = cta_offset + thread_id;
            row_begin = row_offsets[node];
            row_end = row_offsets[node + 1];
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
            KEY_TYPE neighbour;
            KEY_TYPE cur_key;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE block_aggregate;
            while (__syncthreads_or(gather < gather_end)) {
                if (gather < gather_end) {
                    cur_key = (SIZE_TYPE) (keys[gather] & COL_IDX_NONE);
                    VALUE_TYPE cur_value = values[gather];
                    // neighbour = (SIZE_TYPE) (cur_key & COL_IDX_NONE);
                    thread_data_in = ((SIZE_TYPE)cur_key == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : 1;
                    neighbour = ((SIZE_TYPE)cur_key == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : ((keys[gather] & KEY_HIGH) | (KEY_TYPE)label[cur_key]);
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
                row_begin = row_end;
            }

            SIZE_TYPE gather = comm[warp_id][1] + lane_id;
            SIZE_TYPE gather_end = comm[warp_id][2];
            KEY_TYPE neighbour;
            KEY_TYPE cur_key;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE warp_aggregate;
            while (__any_sync(FULL_MASK, gather < gather_end)) {
                if (gather < gather_end) {
                    cur_key = ((SIZE_TYPE) (keys[gather] & COL_IDX_NONE));
                    VALUE_TYPE cur_value = values[gather];
                    thread_data_in = ((SIZE_TYPE)cur_key == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : 1;
                    neighbour = ((SIZE_TYPE)cur_key == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : ((keys[gather] & KEY_HIGH) | (KEY_TYPE)label[cur_key]);
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
                rsv_rank++;
                row_begin++;
            }
            __syncthreads();
            KEY_TYPE neighbour;
            SIZE_TYPE cur_key;
            // gather batch of adjlist
            if (thread_id < min(remain, THREADS_NUM)) {
                cur_key = (SIZE_TYPE) (keys[comm2[thread_id]] & COL_IDX_NONE);
                VALUE_TYPE cur_value = values[comm2[thread_id]];
                thread_data = ((SIZE_TYPE)cur_key == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : 1;
                neighbour = ((SIZE_TYPE)cur_key == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : ((keys[comm2[thread_id]] & KEY_HIGH) | (KEY_TYPE)label[cur_key]);
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

// Initialize all labels at once with GPU.Initially
// each vertex v is assigned a unique label which matches its identifier.
__global__ void gather_cdlp (KEY_TYPE *input, SIZE_TYPE *labels_begin, SIZE_TYPE *labels_end, SIZE_TYPE count) {
    SIZE_TYPE tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > count - 1) {
        return;
    }
    if ((input[tid] & KEY_HIGH) != (input[tid + 1] & KEY_HIGH)) {
        labels_end[(SIZE_TYPE)(input[tid] >> 32)] = labels_begin[(SIZE_TYPE)(input[tid + 1] >> 32)] = tid + 1;
    }
}

__global__ void prop_labels_kernel (KEY_TYPE *prop_labels, SIZE_TYPE *labels_begin, SIZE_TYPE *labels_end, SIZE_TYPE *labels, SIZE_TYPE N) {
    SIZE_TYPE tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= 0 && tid < N) {
        if (labels_begin[tid] >= labels_end[tid]) return;
        SIZE_TYPE maxlabel = (SIZE_TYPE)(prop_labels[labels_begin[tid]] & COL_IDX_NONE), maxcount = 0; // the label that appears the most times and its number of occurrences
        KEY_TYPE last_label = prop_labels[tid];
        for (SIZE_TYPE c = labels_begin[tid], last_count = 0; c < labels_end[tid]; c ++) { // traverse the neighbor vertex label data in order
            if (prop_labels[c] == last_label) {
                last_count++; // add up the number of label occurrences
                if (last_count > maxcount) { // the number of label occurrences currently traversed is greater than the recorded value
                    maxcount = last_count; // update maxcount and maxlabel
                    maxlabel = (SIZE_TYPE)(last_label & COL_IDX_NONE);
                }
            } else {
                last_label = prop_labels[c]; // a new label appears, updates the label and number of occurrences
                last_count = 1;
            }
        }
        labels[tid] = maxlabel; // record the maxlabel
    }
}

__global__ void Label_sum_init (SIZE_TYPE *labels_sum, SIZE_TYPE N) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < N; i += block_offset) {
        labels_sum[i] = 0;
    }
}

__global__ void Label_init_v2 (SIZE_TYPE *labels, SIZE_TYPE N) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < N; i += block_offset) {
        labels[i] = i;
    }
}

__host__ void gpma_cdlp(graph_structure<double> &graph, GPMA& gpma_in, GPMA& gpma_out, std::vector<std::string>& res, int max_iterations) {

    SIZE_TYPE N = graph.size(); // number of vertices in the graph
    const SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM = (N + THREADS_NUM - 1) / THREADS_NUM;

    KEY_TYPE *prop_labels = nullptr;
    SIZE_TYPE *labels = nullptr;
    SIZE_TYPE *labels_begin = nullptr;
    SIZE_TYPE *labels_end = nullptr;
    SIZE_TYPE *node_queue_offset;
    
    size_t free_mem, total_mem;

    int CD_ITERATION = max_iterations; // fixed number of iterations
    long long E_in = gpma_in.get_size(); // number of edges in the graph_in
    long long E_out = gpma_out.get_size(); // number of edges in the graph_out

    SIZE_TYPE *prop_label_offset;
    cudaMalloc(&prop_label_offset, sizeof(SIZE_TYPE));
    cudaMalloc((void**)&labels, (N) * sizeof(SIZE_TYPE));
    cudaMalloc((void**)&labels_begin, (N) * sizeof(SIZE_TYPE));
    cudaMalloc((void**)&labels_end, (N) * sizeof(SIZE_TYPE));
    cudaMalloc((void**)&prop_labels, (E_in + E_out + 1) * sizeof(KEY_TYPE));
    cudaMalloc(&node_queue_offset, sizeof(SIZE_TYPE));
    cudaDeviceSynchronize(); // synchronize, ensure the cudaMalloc is complete
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) { // use the cudaGetLastError to check for possible cudaMalloc errors
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }
    SIZE_TYPE init_label_block = CALC_BLOCKS_NUM_NOLIMIT(THREADS_NUM, N);
    Label_init_v2<<<init_label_block, THREADS_NUM>>>(labels, N); // initialize all labels at once with GPU
    cudaDeviceSynchronize(); // synchronize, ensure the cudaMalloc is complete
    
    size_t temp_storage_bytes = 0;
    void* d_temp_storage = nullptr;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, prop_labels, prop_labels, E_in + E_out);
    cudaDeviceSynchronize();
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "Error: " << "Malloc failed" << " (" << cudaGetErrorString(cuda_status) << ")" << std::endl;
        return;
    }
    
    SIZE_TYPE host_num[1];
    host_num[0] = N;
    cudaMemcpy(node_queue_offset, host_num, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);

    int it = 0;
    float total_time = 0.0f;
    while (it < CD_ITERATION) { // continue for a fixed number of iterations
        BLOCKS_NUM = CALC_BLOCKS_NUM_NOLIMIT(THREADS_NUM, N);
        SIZE_TYPE zero = 0;

        // step 1: label propagation
        cudaMemcpy(prop_label_offset, &zero, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        gpma_cdlp_gather_kernel<THREADS_NUM><<<BLOCKS_NUM, THREADS_NUM>>>(prop_labels, prop_label_offset,
            RAW_PTR(gpma_in.keys), RAW_PTR(gpma_in.values), RAW_PTR(gpma_in.row_offset), labels, node_queue_offset);
        cudaDeviceSynchronize();
        gpma_cdlp_gather_kernel<THREADS_NUM><<<BLOCKS_NUM, THREADS_NUM>>>(prop_labels, prop_label_offset,
            RAW_PTR(gpma_out.keys), RAW_PTR(gpma_out.values), RAW_PTR(gpma_out.row_offset), labels, node_queue_offset);
        cudaDeviceSynchronize();
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) { // use the cudaGetLastError to check for possible cudaMalloc errors
            fprintf(stderr, "Label propagation failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        // step 2: sort
        SIZE_TYPE offset;
        cudaMemcpy(&offset, prop_label_offset, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, prop_labels, prop_labels, offset);
        cudaDeviceSynchronize();
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) { // use the cudaGetLastError to check for possible cudaMalloc errors
            fprintf(stderr, "Sort failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }
        
        // step3: gather
        KEY_TYPE tmp = (((KEY_TYPE)N) << 32) | COL_IDX_NONE;
        cudaMemcpy(prop_labels + offset, &tmp, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        BLOCKS_NUM = CALC_BLOCKS_NUM_NOLIMIT(THREADS_NUM, offset);
        gather_cdlp<<<BLOCKS_NUM, THREADS_NUM>>>(prop_labels, labels_begin, labels_end, offset);
        cudaDeviceSynchronize();
        cudaMemcpy(labels_begin, &zero, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) { // use the cudaGetLastError to check for possible cudaMalloc errors
            fprintf(stderr, "gather failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        // step4: prop label
        BLOCKS_NUM = CALC_BLOCKS_NUM_NOLIMIT(THREADS_NUM, N);
        prop_labels_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(prop_labels, labels_begin, labels_end, labels, N);
        cudaDeviceSynchronize();

        it ++;
    }
    res.resize(N);

    SIZE_TYPE* label_host = new SIZE_TYPE[N];
    cudaMemcpy(label_host, labels, N * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < N; i++) {
        res[i] = graph.vertex_id_to_str[label_host[i]].first; // convert the label to string and store it in res
    }
    
    cudaFree(labels);
    cudaFree(prop_labels);
    cudaFree(labels_begin);
    cudaFree(labels_end);
    cudaFree(node_queue_offset);
    cudaFree(prop_label_offset);
    cudaFree(d_temp_storage);
}

std::vector<std::pair<std::string, std::string>> Cuda_CDLP_optimized(graph_structure<double> &graph, GPMA& gpma_in, GPMA& gpma_out, int max_iterations) {
    std::vector<std::string> result;
    gpma_cdlp(graph, gpma_in, gpma_out, result, max_iterations);
    
    std::vector<std::pair<std::string, std::string>> res;
    int size = result.size();
    for (int i = 0; i < size; i++) {
        res.push_back(std::make_pair(graph.vertex_id_to_str[i].first, result[i])); // for each vertex, get its string number and store it in res
    }

    return res;    
}