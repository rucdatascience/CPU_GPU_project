#pragma once
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
/*for GPU*/
template <typename weight_type>
class CSR_graph {
public:
    std::vector<int> INs_Neighbor_start_pointers, OUTs_Neighbor_start_pointers; // Neighbor_start_pointers[i] is the start point of neighbor information of vertex i in Edges and Edge_weights
    /*
        Now, Neighbor_sizes[i] = Neighbor_start_pointers[i + 1] - Neighbor_start_pointers[i].
        And Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
    */
    std::vector<int> INs_Edges, OUTs_Edges;  // Edges[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] neighbor IDs
    std::vector<weight_type> INs_Edge_weights, OUTs_Edge_weights; // Edge_weights[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] edge weights
    int *in_pointer, *out_pointer, *in_edge, *out_edge;
    double *in_edge_weight,*out_edge_weight;

};

template <typename weight_type>
CSR_graph<weight_type> toCSR(graph_structure<weight_type>& graph) {

    CSR_graph<weight_type> ARRAY;

    int V = graph.size();
    ARRAY.INs_Neighbor_start_pointers.resize(V + 1); // Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
    ARRAY.OUTs_Neighbor_start_pointers.resize(V + 1);

    int pointer = 0;
    for (int i = 0; i < V; i++) {
        ARRAY.INs_Neighbor_start_pointers[i] = pointer;
        for (auto& xx : graph.INs[i]) {
            ARRAY.INs_Edges.push_back(xx.first);
            ARRAY.INs_Edge_weights.push_back(xx.second);
        }
        pointer += graph.INs[i].size();
    }
    ARRAY.INs_Neighbor_start_pointers[V] = pointer;

    pointer = 0;
    for (int i = 0; i < V; i++) {
        ARRAY.OUTs_Neighbor_start_pointers[i] = pointer;
        for (auto& xx : graph.OUTs[i]) {
            ARRAY.OUTs_Edges.push_back(xx.first);
            ARRAY.OUTs_Edge_weights.push_back(xx.second);
        }
        pointer += graph.OUTs[i].size();
    }
    ARRAY.OUTs_Neighbor_start_pointers[V] = pointer;

    int E_in = ARRAY.INs_Edges.size();
    int E_out = ARRAY.OUTs_Edges.size();
    cudaMallocManaged(&ARRAY.in_pointer, (V + 1) * sizeof(int));
    cudaMallocManaged(&ARRAY.out_pointer, (V + 1) * sizeof(int));
    cudaMallocManaged(&ARRAY.in_edge, E_in * sizeof(int));
    cudaMallocManaged(&ARRAY.out_edge, E_out * sizeof(int));
    cudaMallocManaged(&ARRAY.in_edge_weight, E_in * sizeof(double));
    cudaMallocManaged(&ARRAY.out_edge_weight, E_out * sizeof(double));
    cudaMemcpy(ARRAY.in_pointer, ARRAY.INs_Neighbor_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_pointer, ARRAY.OUTs_Neighbor_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.in_edge, ARRAY.INs_Edges.data(), E_in* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_edge, ARRAY.OUTs_Edges.data(), E_out * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.in_edge_weight, ARRAY.INs_Edge_weights.data(), E_in* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_edge_weight, ARRAY.OUTs_Edge_weights.data(), E_out * sizeof(double), cudaMemcpyHostToDevice);

    return ARRAY;
}





