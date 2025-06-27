#ifndef CDLPGPU
#define CDLPGPU

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/cub.cuh>
#include <vector>
#include <string.h>
#include <GPU_gpma/GPU_gpma.hpp>

using namespace std;
#define CD_THREAD_PER_BLOCK 512

__global__ void Label_init(int *labels, int N);
__global__ void LabelPropagation(KEY_TYPE *keys_in, VALUE_TYPE *values_in, SIZE_TYPE *row_offset_in, KEY_TYPE *keys_out, VALUE_TYPE *values_out, SIZE_TYPE *row_offset_out, int *prop_labels, int *labels, int N, long long E_in);
__global__ void Get_New_Label(SIZE_TYPE *row_offset_in, SIZE_TYPE *row_offset_out, int *prop_labels, int *new_labels, int N, long long E_in);
void checkCudaError(cudaError_t err, const char* msg);
void checkDeviceProperties();

void CDLP_GPU(graph_structure<double>& graph, CSR_graph<double>& input_graph, std::vector<string>& res, int max_iterations);

std::vector<std::pair<std::string, std::string>> Cuda_CDLP(graph_structure<double>& graph, CSR_graph<double>& input_graph, int max_iterations);

// propagate the label, the label of the neighbor vertex is propagated in parallel
__global__ void LabelPropagation(KEY_TYPE *keys_in, VALUE_TYPE *values_in, SIZE_TYPE *row_offset_in, KEY_TYPE *keys_out, VALUE_TYPE *values_out, SIZE_TYPE *row_offset_out, int *prop_labels, int *labels, int N, long long E_in) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

    if (tid >= 0 && tid < N) {
        KEY_TYPE new_v;
        VALUE_TYPE new_w;
        
        SIZE_TYPE start = row_offset_in[tid], end = row_offset_in[tid + 1];
        for (int c = start; c < end; c ++) { // traverse the neighbor of the tid vertex
            new_v = keys_in[c] & 0xFFFFFFFF;
            new_w = values_in[c];
            if (new_v != COL_IDX_NONE && new_w != VALUE_NONE) {
                prop_labels[c] = labels[new_v]; // record the label of the neighbor vertex
            } else {
                prop_labels[c] = -1;
            }
        }

        start = row_offset_out[tid], end = row_offset_out[tid + 1];
        for (int c = start; c < end; c ++) { // traverse the neighbor of the tid vertex
            new_v = keys_out[c] & 0xFFFFFFFF;
            new_w = values_out[c];
            if (new_v != COL_IDX_NONE && new_w != VALUE_NONE) {
                prop_labels[E_in + c] = labels[new_v]; // record the label of the neighbor vertex
            } else {
                prop_labels[E_in + c] = -1; // record the label of the neighbor vertex
            }
        }
    }
}

// Initialize all labels at once with GPU.Initially
// each vertex v is assigned a unique label which matches its identifier.
__global__ void Label_init(int *labels, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

    if (tid >= 0 && tid < N) { // tid decides process which vertex
        labels[tid] = tid; // each vertex is initially labeled by itself
    }
}

// each thread is responsible for one vertex
// every segmentation are sorted
// count Frequency from the start in the global_space_for_label to the end in the global_space_for_label
// the new labels are stroed in the new_labels
__global__ void Get_New_Label(SIZE_TYPE *row_offset_in, SIZE_TYPE *row_offset_out, int *prop_labels, int *new_labels, int N, long long E_in) {
    // Use GPU to propagate all labels at the same time.
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // tid decides process which vertex
    if (tid >= 0 && tid < N) {
        int maxlabel = -1, maxcount = 0, last_label = -1, last_count = 0; // the label that appears the most times and its number of occurrences
        SIZE_TYPE start_in = row_offset_in[tid], end_in = row_offset_in[tid + 1];
        SIZE_TYPE start_out = row_offset_out[tid] + E_in, end_out = row_offset_out[tid + 1] + E_in;
        SIZE_TYPE p1 = start_in, p2 = start_out;
        int prop_labels_now;
        while (p1 < end_in || p2 < end_out) {
            if (p1 >= end_in) {
                prop_labels_now = prop_labels[p2 ++];
            } else if (p2 >= end_out) {
                prop_labels_now = prop_labels[p1 ++];
            } else {
                if (prop_labels[p2] > prop_labels[p1]) {
                    prop_labels_now = prop_labels[p1 ++];
                } else {
                    prop_labels_now = prop_labels[p2 ++];
                }
            }
            if (prop_labels_now != -1) {
                if (maxlabel == -1) {
                    maxlabel = prop_labels_now, maxcount = 1;
                    last_label = prop_labels_now, last_count = 1;
                } else {
                    if (prop_labels_now == last_label) {
                        last_count ++; // add up the number of label occurrences
                        if (last_count > maxcount) {// the number of label occurrences currently traversed is greater than the recorded value
                            maxcount = last_count, maxlabel = last_label; // update maxcount and maxlabel
                        }
                    } else {
                        last_label = prop_labels_now, last_count = 1; // a new label appears, updates the label and number of occurrences
                    }
                }
            }
        }
        new_labels[tid] = maxlabel; // record the maxlabel
    }
}

// Community Detection Using Label Propagation on GPU
// Returns label of the graph based on the graph and number of iterations.
void CDLP_GPU(graph_structure<double>& graph, GPMA& gpma_in, GPMA &gpma_out, std::vector<string>& res, int max_iterations) {
    int N = graph.size(); // number of vertices in the graph
    dim3 init_label_block((N + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1); // the number of blocks used in the gpu
    dim3 init_label_thread(CD_THREAD_PER_BLOCK, 1, 1); // the number of threads used in the gpu

    int* prop_labels = nullptr;
    int* new_prop_labels = nullptr;
    int* new_labels = nullptr;
    int* labels = nullptr;

    int CD_ITERATION = max_iterations; // fixed number of iterations
    long long E_in = gpma_in.get_size(); // number of edges in the graph_in
    long long E_out = gpma_out.get_size(); // number of edges in the graph_out

    cudaMallocManaged((void**)&new_labels, N * sizeof(int));
    cudaMallocManaged((void**)&labels, N * sizeof(int));
    cudaMallocManaged((void**)&prop_labels, (E_in + E_out) * sizeof(int));
    cudaMallocManaged((void**)&new_prop_labels, (E_in + E_out) * sizeof(int));

    cudaDeviceSynchronize(); // synchronize, ensure the cudaMalloc is complete
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) { // use the cudaGetLastError to check for possible cudaMalloc errors
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    Label_init<<<init_label_block, init_label_thread>>>(labels, N); // initialize all labels at once with GPU

    cudaDeviceSynchronize(); // synchronize, ensure the label initialization is complete
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) { // use the cudaGetLastError to check for possible label initialization errors
        fprintf(stderr, "Label init failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    int it = 0; // number of iterations
    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortKeys(
        d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels, E_in, N,
        RAW_PTR(gpma_in.row_offset), RAW_PTR(gpma_in.row_offset) + 1); // sort the labels of each vertex's neighbors
    cudaDeviceSynchronize();
    cub::DeviceSegmentedSort::SortKeys(
        d_temp_storage, temp_storage_bytes, prop_labels + E_in, new_prop_labels + E_in, E_out, N,
        RAW_PTR(gpma_out.row_offset), RAW_PTR(gpma_out.row_offset) + 1); // sort the labels of each vertex's neighbors
    cudaDeviceSynchronize();
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Sort failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cerr << "Error: " << "Malloc failed" << " (" << cudaGetErrorString(err) << ")" << endl;
        return;
    }

    while (it < CD_ITERATION) { // continue for a fixed number of iterations
        LabelPropagation<<<init_label_block, init_label_thread>>>(
            RAW_PTR(gpma_in.keys), RAW_PTR(gpma_in.values), RAW_PTR(gpma_in.row_offset),
            RAW_PTR(gpma_out.keys), RAW_PTR(gpma_out.values), RAW_PTR(gpma_out.row_offset), prop_labels, labels, N, E_in); // calculate the neighbor label array for each vertex
        cudaDeviceSynchronize();  // synchronize, ensure the label propagation is complete

        cuda_status = cudaGetLastError(); // check for errors
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "LabelPropagation failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        // Run sorting operation
        cub::DeviceSegmentedSort::SortKeys(
        d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels, E_in, N,
        RAW_PTR(gpma_in.row_offset), RAW_PTR(gpma_in.row_offset) + 1); // sort the labels of each vertex's neighbors
        cudaDeviceSynchronize();
        cub::DeviceSegmentedSort::SortKeys(
        d_temp_storage, temp_storage_bytes, prop_labels + E_in, new_prop_labels + E_in, E_out, N,
        RAW_PTR(gpma_out.row_offset), RAW_PTR(gpma_out.row_offset) + 1); // sort the labels of each vertex's neighbors
        cudaDeviceSynchronize();

        cuda_status = cudaGetLastError(); // check for errors
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Sort failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }
        
        Get_New_Label<<<init_label_block, init_label_thread>>>(
            RAW_PTR(gpma_in.row_offset), RAW_PTR(gpma_out.row_offset), new_prop_labels, new_labels, N, E_in); // generate a new vertex label by label propagation information
        cudaDeviceSynchronize();

        cuda_status = cudaGetLastError(); // check for errors
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Get_New_Label failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        it++; // record number of iterations
        std::swap(labels, new_labels); // store the updated label in the labels
    }
    res.resize(N);
    for (int i = 0; i < N; i++) {
        if (labels[i] == -1) {
            res[i] = graph.vertex_id_to_str[i].first;
        } else {
            res[i] = graph.vertex_id_to_str[labels[i]].first; // convert the label to string and store it in res
        }
    }
    cudaDeviceSynchronize();

    cudaFree(prop_labels); // free memory
    cudaFree(new_prop_labels);
    cudaFree(new_labels);
    cudaFree(d_temp_storage);
    cudaFree(labels);
}

// check whether cuda errors occur and output error information
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << endl; // output error message
        exit(EXIT_FAILURE);
    }
}

// Community Detection Using Label Propagation on GPU
// Returns label of the graph based on the graph and number of iterations.
// the type of the vertex and label are string
std::vector<std::pair<std::string, std::string>> Cuda_CDLP(graph_structure<double>& graph, GPMA& gpma_in, GPMA &gpma_out, int max_iterations) {
    std::vector<std::string> result;
    CDLP_GPU(graph, gpma_in, gpma_out, result, max_iterations); // get the labels of each vertex. vector index is the id of vertex

    std::vector<std::pair<std::string, std::string>> res;
    int size = result.size();
    for (int i = 0; i < size; i++)
        res.push_back(std::make_pair(graph.vertex_id_to_str[i].first, result[i])); // for each vertex, get its string number and store it in res
    
    return res; // return the results
}

#endif