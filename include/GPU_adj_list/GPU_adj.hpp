#pragma once
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>

#include <vector>

#include <CPU_adj_list/CPU_adj_list.hpp>
#include <GPU_adj_list/binary_vector.cuh>

#define RESERVATION_RATE 1.2

template <typename weight_type>
class GPU_adj
{

public:
    GPU_adj() {}
    ~GPU_adj();

    bool is_directed = true;
    int V = 0;
    int max_V = 0;

    std::vector<bool> vertex_valid;

    cuda_vector<cuda_vector<std::pair<int, weight_type>>*>* INs_Edges;
    cuda_vector<cuda_vector<std::pair<int, weight_type>>*>* OUTs_Edges;
    cuda_vector<cuda_vector<std::pair<int, weight_type>>*>* all_Edges;

    cuda_vector<std::pair<int, weight_type>>** in_edge();
    cuda_vector<std::pair<int, weight_type>>** out_edge() { return OUTs_Edges->data(); }
    cuda_vector<std::pair<int, weight_type>>** all_edge();

    void add_edge(int u, int v, weight_type w);
    void remove_edge(int u, int v);
    void remove_all_adjacent_edges(int v);

    void add_vertex(int v);
    void remove_vertex(int v);

};

template <typename weight_type>
void GPU_adj<weight_type>::add_edge(int u, int v, weight_type w)
{
    binary_insert((*OUTs_Edges)[u], v, w);
    if (is_directed) {
        binary_insert((*INs_Edges)[v], u, w);
        binary_insert((*all_Edges)[u], v, w);
        binary_insert((*all_Edges)[v], u, w);
    }
    else
        binary_insert((*OUTs_Edges)[v], u, w);
}

template <typename weight_type>
void GPU_adj<weight_type>::remove_edge(int u, int v)
{
    binary_earse((*OUTs_Edges)[u], v);
    if (is_directed) {
        binary_earse((*INs_Edges)[v], u);
        binary_earse((*all_Edges)[u], v);
        binary_earse((*all_Edges)[v], u);
    }
    else
        binary_earse((*OUTs_Edges)[v], u);
}

template <typename weight_type>
void GPU_adj<weight_type>::remove_all_adjacent_edges(int v)
{
    if (is_directed) {
        size_t size = (*OUTs_Edges)[v]->size();
        for (size_t i = 0; i < size; i++) {
            binary_earse((*INs_Edges)[(*OUTs_Edges)[v]->operator[](i).first], v);
            binary_earse((*all_Edges)[(*OUTs_Edges)[v]->operator[](i).first], v);
        }

        size = (*INs_Edges)[v]->size();
        for (size_t i = 0; i < size; i++) {
            binary_earse((*OUTs_Edges)[(*INs_Edges)[v]->operator[](i).first], v);
            binary_earse((*all_Edges)[(*INs_Edges)[v]->operator[](i).first], v);
        }

        (*OUTs_Edges)[v]->clear();
        (*INs_Edges)[v]->clear();
        (*all_Edges)[v]->clear();
    }
    else {
        size_t size = (*OUTs_Edges)[v]->size();
        for (size_t i = 0; i < size; i++)
            binary_earse((*OUTs_Edges)[(*OUTs_Edges)[v]->operator[](i).first], v);

        (*OUTs_Edges)[v]->clear();
    }
}

template <typename weight_type>
void GPU_adj<weight_type>::add_vertex(int v) {
    if (v < max_V) {
        vertex_valid[v] = true;
        return;
    }

    if (v > max_V) {
        std::cout << "Error: add_vertex v > max_V" << std::endl;
        return;
    }

    max_V++;
    vertex_valid.push_back(true);

    cuda_vector<std::pair<int, weight_type>>* x;
    cudaMallocManaged(&x, sizeof(cuda_vector<std::pair<int, weight_type>>));
    new (x) cuda_vector<std::pair<int, weight_type>>();
    OUTs_Edges->push_back(x);

    if (is_directed) {
        cuda_vector<std::pair<int, weight_type>>* y;
        cudaMallocManaged(&y, sizeof(cuda_vector<std::pair<int, weight_type>>));
        new (y) cuda_vector<std::pair<int, weight_type>>();
        INs_Edges->push_back(y);

        cuda_vector<std::pair<int, weight_type>>* z;
        cudaMallocManaged(&z, sizeof(cuda_vector<std::pair<int, weight_type>>));
        new (z) cuda_vector<std::pair<int, weight_type>>();
        all_Edges->push_back(z);
    }
}

template <typename weight_type>
void GPU_adj<weight_type>::remove_vertex(int v) {
    if (v >= max_V) {
        std::cout << "Error: remove_vertex v >= max_V" << std::endl;
        return;
    }

    vertex_valid[v] = false;

    remove_all_adjacent_edges(v);
}

template <typename weight_type>
GPU_adj<weight_type> to_GPU_adj(graph_structure<weight_type>& graph, bool is_directed = true)
{

    GPU_adj<weight_type> gpu_adj;

    gpu_adj.is_directed = is_directed;

    gpu_adj.V = graph.size();
    gpu_adj.max_V = gpu_adj.V;

    gpu_adj.vertex_valid.resize(gpu_adj.max_V, true);

    cudaMallocManaged(&gpu_adj.OUTs_Edges, sizeof(cuda_vector<cuda_vector<std::pair<int, weight_type>>*>));
    cudaDeviceSynchronize();
    new (gpu_adj.OUTs_Edges) cuda_vector<cuda_vector<std::pair<int, weight_type>>*>(gpu_adj.V);
    gpu_adj.OUTs_Edges->resize(gpu_adj.V);

    for (int i = 0; i < gpu_adj.V; i++) {
        cudaMallocManaged(&((*gpu_adj.OUTs_Edges)[i]), sizeof(cuda_vector<std::pair<int, weight_type>>));
        unsigned long long init_capacity = graph.OUTs[i].size() * RESERVATION_RATE;
        new ((*gpu_adj.OUTs_Edges)[i]) cuda_vector<std::pair<int, weight_type>>(init_capacity);
    }

    if (is_directed) {

        cudaMallocManaged(&gpu_adj.INs_Edges, sizeof(cuda_vector<cuda_vector<std::pair<int, weight_type>>*>));
        cudaDeviceSynchronize();
        new (gpu_adj.INs_Edges) cuda_vector<cuda_vector<std::pair<int, weight_type>>*>(gpu_adj.V);
        gpu_adj.INs_Edges->resize(gpu_adj.V);
        for (int i = 0; i < gpu_adj.V; i++) {
            cudaMallocManaged(&((*gpu_adj.INs_Edges)[i]), sizeof(cuda_vector<std::pair<int, weight_type>>));
            unsigned long long init_capacity = graph.INs[i].size() * RESERVATION_RATE;
            new ((*gpu_adj.INs_Edges)[i]) cuda_vector<std::pair<int, weight_type>>(init_capacity);
        }

        for (int i = 0; i < gpu_adj.V; i++) {
            for (auto& xx : graph.INs[i])
                binary_insert((*gpu_adj.INs_Edges)[i], xx.first, xx.second);
        }

        cudaMallocManaged(&gpu_adj.all_Edges, sizeof(cuda_vector<cuda_vector<std::pair<int, weight_type>>*>));
        cudaDeviceSynchronize();
        new (gpu_adj.all_Edges) cuda_vector<cuda_vector<std::pair<int, weight_type>>*>(gpu_adj.V);
        gpu_adj.all_Edges->resize(gpu_adj.V);
        for (int i = 0; i < gpu_adj.V; i++) {
            cudaMallocManaged(&((*gpu_adj.all_Edges)[i]), sizeof(cuda_vector<std::pair<int, weight_type>>));
            unsigned long long init_capacity = (graph.INs[i].size() + graph.OUTs[i].size()) * RESERVATION_RATE; 
            new ((*gpu_adj.all_Edges)[i]) cuda_vector<std::pair<int, weight_type>>(init_capacity);
        }

        for (int i = 0; i < gpu_adj.V; i++) {
            for (auto& xx : graph.INs[i])
                binary_insert((*gpu_adj.all_Edges)[i], xx.first, xx.second);
        }
        for (int i = 0; i < gpu_adj.V; i++) {
            for (auto& xx : graph.OUTs[i])
                binary_insert((*gpu_adj.all_Edges)[i], xx.first, xx.second);
        }
    }

    for (int i = 0; i < gpu_adj.V; i++) {
        for (auto& xx : graph.OUTs[i])
            binary_insert((*gpu_adj.OUTs_Edges)[i], xx.first, xx.second);
    }

    return gpu_adj;
}

template <typename weight_type>
cuda_vector<std::pair<int, weight_type>>** GPU_adj<weight_type>::in_edge()
{
    if (!is_directed) {
        return OUTs_Edges->data();
    }
    return INs_Edges->data();
}

template <typename weight_type>
cuda_vector<std::pair<int, weight_type>>** GPU_adj<weight_type>::all_edge()
{
    if (!is_directed) {
        return OUTs_Edges->data();
    }
    return all_Edges->data();
}

template <typename weight_type>
GPU_adj<weight_type>::~GPU_adj()
{
    for (int i = 0; i < V; i++) {
        (*OUTs_Edges)[i]->~cuda_vector();
        if (is_directed) {
            (*INs_Edges)[i]->~cuda_vector();
            (*all_Edges)[i]->~cuda_vector();
        }
    }

    OUTs_Edges->~cuda_vector();
    cudaFree(OUTs_Edges);
    if (is_directed) {
        INs_Edges->~cuda_vector();
        all_Edges->~cuda_vector();
        cudaFree(INs_Edges);
        cudaFree(all_Edges);
    }
}
