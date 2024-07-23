#ifndef CDLPGPU
#define CDLPGPU

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <vector>
#include <string.h>
#include <csr_graph.hpp>


using namespace std;
#define CD_THREAD_PER_BLOCK 512

__global__ void Label_init(int *labels, int *all_pointer, int N);
__global__ void LabelPropagation(int *all_pointer, int *prop_labels, int *labels, int *all_edge, int N);
__global__ void Get_New_Label(int *all_pointer, int *prop_labels, int *new_labels,  int N);
void checkCudaError(cudaError_t err, const char* msg);
void checkDeviceProperties();

// int gpu_Community_Detection(graph_structure<double> & graph, float* elapsedTime,vector<int> &ans);
void CDLP_GPU(graph_structure<double>& graph, CSR_graph<double>& input_graph, std::vector<string>& res, int max_iterations);

std::vector<std::pair<std::string, std::string>> Cuda_CDLP(graph_structure<double>& graph, CSR_graph<double>& input_graph, int max_iterations);

#endif