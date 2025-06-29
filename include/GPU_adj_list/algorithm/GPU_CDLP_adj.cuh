#ifndef CDLP_GPU_ADJ
#define CDLP_GPU_ADJ

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/cub.cuh>
#include <vector>
#include <string.h>
#include <GPU_adj_list/GPU_adj.hpp>

using namespace std;
#define CD_THREAD_PER_BLOCK 512

__global__ void Label_init(int *labels, int N);
__global__ void LabelPropagation(int *all_pointer, int *prop_labels, int *labels, cuda_vector<std::pair<int, double>> **all_edge, int N);
__global__ void Get_New_Label(int *all_pointer, int *prop_labels, int *new_labels,  int N);
void checkCudaError(cudaError_t err, const char* msg);
void checkDeviceProperties();

void CDLP_GPU(graph_structure<double>& graph, GPU_adj<double>& input_graph, std::vector<string>& res, int max_iterations);

std::vector<std::pair<std::string, std::string>> Cuda_CDLP_adj(graph_structure<double>& graph, GPU_adj<double>& input_graph, int max_iterations);

// propagate the label, the label of the neighbor vertex is propagated in parallel
__global__ void LabelPropagation(int *all_pointer, int *prop_labels, int *labels, cuda_vector<int> **all_edge, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

    if (tid >= 0 && tid < N) {
        int all_edge_offset = all_pointer[tid];
        cuda_vector<int>& ed = *all_edge[tid];
        for (int c = all_edge_offset; c < all_pointer[tid + 1]; c++) { // traverse the neighbor of the tid vertex
            prop_labels[c] = labels[ed[c - all_edge_offset]]; // record the label of the neighbor vertex
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
__global__ void Get_New_Label(int *all_pointer, int *prop_labels, int *new_labels, int N) {
    // Use GPU to propagate all labels at the same time.
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // tid decides process which vertex
    if (tid >= 0 && tid < N) {
        int maxlabel = prop_labels[all_pointer[tid]], maxcount = 0; // the label that appears the most times and its number of occurrences
        for (int c = all_pointer[tid], last_label = prop_labels[all_pointer[tid]], last_count = 0; c < all_pointer[tid + 1]; c++) { // traverse the neighbor vertex label data in order
            if (prop_labels[c] == last_label) {
                last_count++; // add up the number of label occurrences
                if (last_count > maxcount) { // the number of label occurrences currently traversed is greater than the recorded value
                    maxcount = last_count; // update maxcount and maxlabel
                    maxlabel = last_label;
                }
            } else {
                last_label = prop_labels[c]; // a new label appears, updates the label and number of occurrences
                last_count = 1;
            }
        }
        new_labels[tid] = maxlabel; // record the maxlabel
    }
}

// Community Detection Using Label Propagation on GPU
// Returns label of the graph based on the graph and number of iterations.
void CDLP_GPU(graph_structure<double>& graph, GPU_adj<double>& input_graph, std::vector<string>& res, int max_iterations) {
    int N = graph.size(); // number of vertices in the graph
    dim3 init_label_block((N + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1); // the number of blocks used in the gpu
    dim3 init_label_thread(CD_THREAD_PER_BLOCK, 1, 1); // the number of threads used in the gpu

    auto all_edge = input_graph.all_edge(); // graph stored in csr format

    int* all_pointer = nullptr;
    int* prop_labels = nullptr;
    int* new_prop_labels = nullptr;
    int* new_labels = nullptr;
    int* labels = nullptr;

    int CD_ITERATION = max_iterations; // fixed number of iterations
    long long E = 0; // number of edges in the graph
    cudaMallocManaged((void**)&all_pointer, (N + 1) * sizeof(int));
    cudaMallocManaged((void**)&new_labels, N * sizeof(int));
    cudaMallocManaged((void**)&labels, N * sizeof(int));

    cudaDeviceSynchronize(); // synchronize, ensure the cudaMalloc is complete
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) { // use the cudaGetLastError to check for possible cudaMalloc errors
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    for (int i = 0; i < N; i++) {
        all_pointer[i] = E;
        E += all_edge[i]->size();
    }
    all_pointer[N] = E;
    
    cudaMallocManaged((void**)&prop_labels, E * sizeof(int));
    cudaMallocManaged((void**)&new_prop_labels, E * sizeof(int));

    cudaDeviceSynchronize(); // synchronize, ensure the cudaMalloc is complete
    cuda_status = cudaGetLastError();
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
        d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels,
        E, N, all_pointer, all_pointer + 1); // sort the labels of each vertex's neighbors

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
        LabelPropagation<<<init_label_block, init_label_thread>>>(all_pointer, prop_labels, labels, all_edge, N); // calculate the neighbor label array for each vertex
        cudaDeviceSynchronize();  // synchronize, ensure the label propagation is complete

        cuda_status = cudaGetLastError(); // check for errors
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "LabelPropagation failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        // Run sorting operation
        cub::DeviceSegmentedSort::SortKeys(
            d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels,
            E, N, all_pointer, all_pointer + 1); // sort the labels of each vertex's neighbors
        cudaDeviceSynchronize();

        cuda_status = cudaGetLastError(); // check for errors
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Sort failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }
        
        Get_New_Label<<<init_label_block, init_label_thread>>>(all_pointer, new_prop_labels, new_labels,  N); // generate a new vertex label by label propagation information

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
        res[i] = graph.vertex_id_to_str[labels[i]].first; // convert the label to string and store it in res
    }

    cudaFree(labels);
    cudaFree(prop_labels); // free memory
    cudaFree(new_prop_labels);
    cudaFree(new_labels);
    cudaFree(d_temp_storage);
    cudaFree(all_pointer);
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
std::vector<std::pair<std::string, std::string>> Cuda_CDLP_adj(graph_structure<double>& graph, GPU_adj<double>& input_graph, int max_iterations) {
    std::vector<std::string> result;
    CDLP_GPU(graph, input_graph, result, max_iterations); // get the labels of each vertex. vector index is the id of vertex

    std::vector<std::pair<std::string, std::string>> res;
    int size = result.size();
    for (int i = 0; i < size; i++)
        res.push_back(std::make_pair(graph.vertex_id_to_str[i].first, result[i])); // for each vertex, get its string number and store it in res
    
    return res; // return the results
}

#endif