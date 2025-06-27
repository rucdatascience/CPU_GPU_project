#pragma once
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <CPU_adj_list/CPU_adj_list.hpp>
/*for GPU*/
template <typename weight_type>
class CSR_graph
{
    //CSR has space efficiency and is easy to use by GPUs.
public:
    CSR_graph() {}
    ~CSR_graph();

    bool is_directed = true; // direct graph or undirect graph

    std::vector<int> INs_Neighbor_start_pointers, OUTs_Neighbor_start_pointers, ALL_start_pointers; // Neighbor_start_pointers[i] is the start point of neighbor information of vertex i in Edges and Edge_weights
    /*
        Now, Neighbor_sizes[i] = Neighbor_start_pointers[i + 1] - Neighbor_start_pointers[i].
        And Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
    */
    std::vector<int> INs_Edges, OUTs_Edges, all_Edges; // Edges[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] neighbor IDs
    std::vector<weight_type> INs_Edge_weights, OUTs_Edge_weights; // Edge_weights[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] edge weights
    int *in_pointer, *out_pointer, *in_edge, *out_edge, *all_pointer, *all_edge;//All_edge has merged in_edge and out_edge, mainly used on CDLP
    double *in_edge_weight, *out_edge_weight;
    size_t E_all;
};

template <typename weight_type>
// CSR_graph<weight_type> toCSR(graph_structure<weight_type>& graph)
CSR_graph<weight_type> toCSR(graph_structure<weight_type>& graph, bool is_directed = true)
{
    CSR_graph<weight_type> ARRAY;

    ARRAY.is_directed = is_directed;

    int V = graph.size();
    ARRAY.OUTs_Neighbor_start_pointers.resize(V + 1);

    int pointer = 0;

    if (is_directed) {
        ARRAY.INs_Neighbor_start_pointers.resize(V + 1); // Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
        for (int i = 0; i < V; i++)
        {
            ARRAY.INs_Neighbor_start_pointers[i] = pointer;
            for (auto &xx : graph.INs[i])
            {
                ARRAY.INs_Edges.push_back(xx.first);
                ARRAY.INs_Edge_weights.push_back(xx.second);
            }
            pointer += graph.INs[i].size();
        }
        ARRAY.INs_Neighbor_start_pointers[V] = pointer;
    }

    pointer = 0;
    for (int i = 0; i < V; i++)
    {
        ARRAY.OUTs_Neighbor_start_pointers[i] = pointer;
        for (auto &xx : graph.OUTs[i])
        {
            ARRAY.OUTs_Edges.push_back(xx.first);
            ARRAY.OUTs_Edge_weights.push_back(xx.second);
        }
        pointer += graph.OUTs[i].size();
    }
    ARRAY.OUTs_Neighbor_start_pointers[V] = pointer;

    if (is_directed) {
        ARRAY.ALL_start_pointers.resize(V + 1);
        pointer = 0;
        for (int i = 0; i < V; i++)
        {
            ARRAY.ALL_start_pointers[i] = pointer;
            for (auto &xx : graph.INs[i])
            {
                ARRAY.all_Edges.push_back(xx.first);
            }
            for (auto &xx : graph.OUTs[i])
            {
                ARRAY.all_Edges.push_back(xx.first);
            }
            pointer += graph.INs[i].size() + graph.OUTs[i].size();
        }
        ARRAY.ALL_start_pointers[V] = pointer;
    }

    size_t E_in = 0;
    size_t E_out = ARRAY.OUTs_Edges.size();
    size_t E_all = E_out;
    
    ARRAY.E_all = E_all;

    cudaMalloc((void**)&ARRAY.out_pointer, (V + 1) * sizeof(int));
    cudaMalloc((void**)&ARRAY.out_edge, E_out * sizeof(int));
    cudaMalloc((void**)&ARRAY.out_edge_weight, E_out * sizeof(double));

    if (is_directed) {
        E_in = ARRAY.INs_Edges.size();
        E_all = E_in + E_out;
        ARRAY.E_all = E_all;

        cudaMalloc((void**)&ARRAY.in_pointer, (V + 1) * sizeof(int));
        cudaMalloc((void**)&ARRAY.all_pointer, (V + 1) * sizeof(int));
        cudaMalloc((void**)&ARRAY.in_edge, E_in * sizeof(int));
        cudaMalloc((void**)&ARRAY.all_edge, E_all * sizeof(int));
        cudaMalloc((void**)&ARRAY.in_edge_weight, E_in * sizeof(double));
    }
    else {
        ARRAY.in_pointer = ARRAY.out_pointer;
        ARRAY.all_pointer = ARRAY.out_pointer;
        ARRAY.in_edge = ARRAY.out_edge;
        ARRAY.all_edge = ARRAY.out_edge;
        ARRAY.in_edge_weight = ARRAY.out_edge_weight;
    }

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(ARRAY.out_pointer, ARRAY.OUTs_Neighbor_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_edge, ARRAY.OUTs_Edges.data(), E_out * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_edge_weight, ARRAY.OUTs_Edge_weights.data(), E_out * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    if (is_directed) {
        cudaMemcpy(ARRAY.in_pointer, ARRAY.INs_Neighbor_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(ARRAY.all_pointer, ARRAY.ALL_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(ARRAY.in_edge, ARRAY.INs_Edges.data(), E_in * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(ARRAY.all_edge, ARRAY.all_Edges.data(), E_all * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(ARRAY.in_edge_weight, ARRAY.INs_Edge_weights.data(), E_in * sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    return ARRAY;
}

template <typename weight_type>
CSR_graph<weight_type>::~CSR_graph()
{
    cudaFree(out_pointer);
    cudaFree(out_edge);
    cudaFree(out_edge_weight);

    if (is_directed) {
        cudaFree(in_pointer);
        cudaFree(in_edge);
        cudaFree(in_edge_weight);

        cudaFree(all_pointer);
        cudaFree(all_edge);
    }
}