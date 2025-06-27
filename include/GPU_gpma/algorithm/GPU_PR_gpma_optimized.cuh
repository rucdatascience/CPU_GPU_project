#pragma once

#include "cub/cub.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <GPU_gpma/GPU_gpma.hpp>
#include <GPU_csr/GPU_csr.hpp>

#define PR_THRES 0.001
#define THREAD_PER_BLOCK 512

template<SIZE_TYPE VECTORS_PER_BLOCK, SIZE_TYPE THREADS_PER_VECTOR>
__global__ void gpma_csr_spmv_pr_kernel(SIZE_TYPE *row_offset, KEY_TYPE *keys, VALUE_TYPE *values,
        SIZE_TYPE row_num, double *x, double *y, double *lp, double q, double s) {

    __shared__  volatile double reduce_data[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
    __shared__  volatile SIZE_TYPE ptrs[VECTORS_PER_BLOCK][2];

    const SIZE_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const SIZE_TYPE thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    const SIZE_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    const SIZE_TYPE vector_id = thread_id / THREADS_PER_VECTOR;
    const SIZE_TYPE vector_lane = threadIdx.x / THREADS_PER_VECTOR;
    const SIZE_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;

    for (SIZE_TYPE row = vector_id; row < row_num; row += num_vectors) {
        // use two threads to fetch row pointer
        if (thread_lane < 2) {
            ptrs[vector_lane][thread_lane] = row_offset[row + thread_lane];
        }

        const SIZE_TYPE row_start = ptrs[vector_lane][0];
        const SIZE_TYPE row_end = ptrs[vector_lane][1];

        double sum = 0.0;

        for (SIZE_TYPE i = row_start + thread_lane; i < row_end; i += THREADS_PER_VECTOR) {
            SIZE_TYPE col_idx = keys[i] & COL_IDX_NONE;
            VALUE_TYPE value = values[i];
            if (COL_IDX_NONE != col_idx && value != VALUE_NONE)
                sum += x[col_idx] / lp[col_idx];
        }

        reduce_data[threadIdx.x] = sum;

        // reduce the sum of threads
        double temp;
        if (THREADS_PER_VECTOR > 16) {
            temp = reduce_data[threadIdx.x + 16];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }
        if (THREADS_PER_VECTOR > 8) {
            temp = reduce_data[threadIdx.x + 8];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }
        if (THREADS_PER_VECTOR > 4) {
            temp = reduce_data[threadIdx.x + 4];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }
        if (THREADS_PER_VECTOR > 2) {
            temp = reduce_data[threadIdx.x + 2];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }
        if (THREADS_PER_VECTOR > 1) {
            temp = reduce_data[threadIdx.x + 1];
            reduce_data[threadIdx.x] = sum = sum + temp;
        }

        // write back answer
        if (0 == thread_lane) {
            y[row] = q * reduce_data[threadIdx.x] + s;
        }
    }
}

// y = p * A * x + (1 - q)
template<SIZE_TYPE THREADS_PER_VECTOR>
__host__ void _pagerank_one(SIZE_TYPE *row_offset, KEY_TYPE *keys, VALUE_TYPE *values,
        SIZE_TYPE row_num, double *lp, double q, double s, double *x, double *y) {
    const SIZE_TYPE THREADS_NUM = 128;
    const SIZE_TYPE VECTORS_PER_BLOCK = THREADS_NUM / THREADS_PER_VECTOR;
    const SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(VECTORS_PER_BLOCK, row_num);

    gpma_csr_spmv_pr_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<BLOCKS_NUM, THREADS_NUM>>>(
            row_offset, keys, values, row_num, x, y, lp, q, s);
}
__host__
void pagerank_one_iteration(SIZE_TYPE *row_offset, KEY_TYPE *keys,
        VALUE_TYPE *values, SIZE_TYPE row_num, double *lp, double q, double s,
        double *x, double *y, SIZE_TYPE avg_nnz_per_row) {

    if (avg_nnz_per_row <= 2) {
        _pagerank_one<2>(row_offset, keys, values, row_num, lp, q, s, x, y);
        return;
    }
    if (avg_nnz_per_row <= 4) {
        _pagerank_one<4>(row_offset, keys, values, row_num, lp, q, s, x, y);
        return;
    }
    if (avg_nnz_per_row <= 8) {
        _pagerank_one<8>(row_offset, keys, values, row_num, lp, q, s, x, y);
        return;
    }
    if (avg_nnz_per_row <= 16) {
        _pagerank_one<16>(row_offset, keys, values, row_num, lp, q, s, x, y);
        return;
    }
    _pagerank_one<32>(row_offset, keys, values, row_num, lp, q, s, x, y);
}

// Implementing atomic operations,
// that is, ensuring that adding operations to a specific
// memory location in a multi-threaded environment are thread safe.
__device__ double _atomicAdd_v2(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// A reduction pattern was used to sum up the sink value
__global__ void calculate_sink_v2(double *pr, int *N_out_zero_gpu, int out_zero_size, double *sink_sum) {
    extern __shared__ double sink[]; // Declare shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stid = threadIdx.x;

    if (tid < out_zero_size)
        sink[stid] = pr[N_out_zero_gpu[tid]]; // get PR(w)
    else
        sink[stid] = 0; // not the out-degree zero vertex
    __syncthreads(); // wait unitl finish Loading data into shared memory

    for (int i = blockDim.x / 2; i > 0; i >>= 1) { // get the sum of sink by reducing kernel function
        if (stid < i)
            sink[stid] += sink[stid + i];
        __syncthreads(); // Synchronize again to ensure that each step of the reduction operation is completed
    }
    if (stid == 0)
        _atomicAdd_v2(sink_sum, sink[0]); // Write the result of each thread block into the output array
}

__global__ void get_sink_vertex(SIZE_TYPE *row_offset, KEY_TYPE *keys, VALUE_TYPE *values, double *d_lp, SIZE_TYPE row_num) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < row_num) {
        for (int j = row_offset[tid]; j < row_offset[tid + 1]; j++) {
            KEY_TYPE key = keys[j];
            VALUE_TYPE value = values[j];
            SIZE_TYPE col_idx = key & COL_IDX_NONE;
            if (COL_IDX_NONE != col_idx && value != VALUE_NONE) {
                d_lp[tid] += 1.0;
            }
        }
    }
}

template<typename T>
struct square {
    __host__  __device__ T operator()(const T& x) const {
        return x * x;
    }
};
template<typename T>
struct ABS {
    __host__  __device__ T operator()(const T& x) const {
        return abs(x);
    }
};

__host__ std::vector<double> gpma_pr(graph_structure<double> &graph, DEV_VEC_KEY &keys, DEV_VEC_VALUE &values, DEV_VEC_SIZE &row_offset, SIZE_TYPE row_num,
        SIZE_TYPE nnz_num, int iterations, double damping) {
    thrust::device_vector<double> pr[2];
    pr[0].resize(row_num, 1.0 / row_num);
    pr[1].resize(row_num, 1.0 / row_num);
    cudaDeviceSynchronize();

    // generate lp array
    thrust::host_vector<double> h_lp(row_num);
    thrust::device_vector<double> d_lp;
    cudaDeviceSynchronize();
    
    dim3 blockPerGrid, threadPerGrid;
    blockPerGrid.x = (row_num + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK; // the number of blocks used in the gpu
    threadPerGrid.x = THREAD_PER_BLOCK; // the number of threads used in the gpu
    
    double* sink_sum = nullptr;
    cudaMallocManaged(&sink_sum, sizeof(double));
    int* sink_vertex_gpu = nullptr;
    std::vector<int> sink_vertexs;
    for (int i = 0; i < row_num; i++) {
        h_lp[i] = graph.out_degree(i);
        if (h_lp[i] == 0) sink_vertexs.push_back(i);
    }
    int out_zero_size = sink_vertexs.size(); // the number of sink vertices
    cudaMallocManaged(&sink_vertex_gpu, out_zero_size * sizeof(int));
    cudaDeviceSynchronize();
    cudaMemcpy(sink_vertex_gpu, sink_vertexs.data(), out_zero_size * sizeof(int), cudaMemcpyHostToDevice);
    d_lp = h_lp;
    cudaDeviceSynchronize();

    SIZE_TYPE avg_nnz_per_row = nnz_num / row_num;

    // double norm2;
    int iteration = 0;

    double teleport = (1 - damping) / row_num;
    while (iteration < iterations) {
        *sink_sum = 0;
        calculate_sink_v2<<<blockPerGrid, threadPerGrid, THREAD_PER_BLOCK * sizeof(double)>>>(RAW_PTR(pr[iteration % 2]), 
                sink_vertex_gpu, out_zero_size, sink_sum); // calculate the sinksum
        cudaDeviceSynchronize();
        
        *sink_sum = (*sink_sum) * damping / row_num;
        pagerank_one_iteration(RAW_PTR(row_offset), RAW_PTR(keys), RAW_PTR(values), row_num, RAW_PTR(d_lp), damping, teleport + (*sink_sum),
                RAW_PTR(pr[iteration % 2]), RAW_PTR(pr[(iteration + 1) % 2]), avg_nnz_per_row);

        iteration ++;
    }
    std::vector<double> results(row_num);
    cudaMemcpy(results.data(), RAW_PTR(pr[(iteration % 2)]), row_num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(sink_sum);
    cudaFree(sink_vertex_gpu);
    d_lp.clear();
    d_lp.shrink_to_fit();
    pr[0].clear();
    pr[0].shrink_to_fit();
    pr[1].clear();
    pr[1].shrink_to_fit();
    return results;
}

std::vector<std::pair<std::string, double>> Cuda_PR_optimized(graph_structure<double> &graph, GPMA& gpma_in, GPMA &gpma_out, int iterations, double damping){
    int V = gpma_out.row_num;
    int E = gpma_out.get_size();
    std::vector<double> prVecGPU = gpma_pr(graph, gpma_in.keys, gpma_in.values, gpma_in.row_offset, V, E, iterations, damping);
    return graph.res_trans_id_val(prVecGPU); // return the results in string type
}