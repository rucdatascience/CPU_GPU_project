#pragma once
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>

#include <vector>

#include <CPU_adj_list/CPU_adj_list.hpp>
#include <GPU_adj_list/binary_vector.cuh>

#define RESERVATION_RATE 2.2

template <typename weight_type>
class GPU_adj {
public:
    GPU_adj() {}
    ~GPU_adj();

    bool is_directed = true;
    int V = 0;
    int max_V = 0;

    std::vector<bool> vertex_valid;

    cuda_vector<cuda_vector<int>*>* INs_Edges;
    cuda_vector<cuda_vector<int>*>* OUTs_Edges;
    cuda_vector<cuda_vector<int>*>* all_Edges;

    cuda_vector<cuda_vector<weight_type>*>* INs_Edges_weight;
    cuda_vector<cuda_vector<weight_type>*>* OUTs_Edges_weight;
    cuda_vector<cuda_vector<weight_type>*>* all_Edges_weight;

    cuda_vector<int>** in_edge() { return (!is_directed) ? OUTs_Edges->data() : INs_Edges->data(); }
    cuda_vector<int>** out_edge() { return OUTs_Edges->data(); }
    cuda_vector<int>** all_edge() { return (!is_directed) ? OUTs_Edges->data() : all_Edges->data(); }

    cuda_vector<weight_type>** in_edge_weight() { return (!is_directed) ? OUTs_Edges_weight->data() : INs_Edges_weight->data(); }
    cuda_vector<weight_type>** out_edge_weight() { return OUTs_Edges_weight->data(); }
    cuda_vector<weight_type>** all_edge_weight() { return (!is_directed) ? OUTs_Edges_weight->data() : all_Edges_weight->data(); };

    void add_edge(int u, int v, weight_type w);
    void remove_edge(int u, int v);
    void remove_all_adjacent_edges(int v);

    void add_vertex(int v);
    void remove_vertex(int v);

};

template <typename weight_type>
void GPU_adj<weight_type>::add_edge(int u, int v, weight_type w) {
    binary_insert((*OUTs_Edges)[u], (*OUTs_Edges_weight)[u], v, w);
    if (is_directed) {
        binary_insert((*INs_Edges)[v], (*INs_Edges_weight)[v], u, w);
        binary_insert((*all_Edges)[u], (*all_Edges_weight)[u], v, w);
        binary_insert((*all_Edges)[v], (*all_Edges_weight)[v], u, w);
    } else {
        binary_insert((*OUTs_Edges)[v], (*OUTs_Edges_weight)[v], u, w);
    }
}

template <typename weight_type>
void GPU_adj<weight_type>::remove_edge(int u, int v) {
    binary_earse((*OUTs_Edges)[u], (*OUTs_Edges_weight)[u], v);
    if (is_directed) {
        binary_earse((*INs_Edges)[v], (*INs_Edges_weight)[v], u);
        binary_earse((*all_Edges)[u], (*all_Edges_weight)[u], v);
        binary_earse((*all_Edges)[v], (*all_Edges_weight)[v], u);
    } else {
        binary_earse((*OUTs_Edges)[v], (*OUTs_Edges_weight)[v], u);
    }
}

template <typename weight_type>
void GPU_adj<weight_type>::remove_all_adjacent_edges(int v) {
    if (is_directed) {
        size_t size = (*OUTs_Edges)[v]->size();
        for (size_t i = 0; i < size; i++) {
            int x = (*OUTs_Edges)[v]->operator[](i);
            binary_earse((*INs_Edges)[x], (*INs_Edges_weight)[x], v);
            binary_earse((*all_Edges)[x], (*all_Edges_weight)[x], v);
        }

        size = (*INs_Edges)[v]->size();
        for (size_t i = 0; i < size; i++) {
            int x = (*INs_Edges)[v]->operator[](i);
            binary_earse((*OUTs_Edges)[x], (*OUTs_Edges_weight)[x], v);
            binary_earse((*all_Edges)[x], (*all_Edges_weight)[x], v);
        }

        (*OUTs_Edges)[v]->clear(), (*OUTs_Edges_weight)[v]->clear();
        (*INs_Edges)[v]->clear(), (*INs_Edges_weight)[v]->clear();
        (*all_Edges)[v]->clear(), (*all_Edges_weight)[v]->clear();
    } else {
        size_t size = (*OUTs_Edges)[v]->size();
        for (size_t i = 0; i < size; i++) {
            int x = (*OUTs_Edges)[v]->operator[](i);
            binary_earse((*OUTs_Edges)[x], (*OUTs_Edges_weight)[x], v);
        }
        (*OUTs_Edges)[v]->clear(), (*OUTs_Edges_weight)[v]->clear();
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

    cuda_vector<int>* x;
    cuda_vector<weight_type>* x_weight;
    cudaMallocManaged(&x, sizeof(cuda_vector<int>));
    cudaMallocManaged(&x_weight, sizeof(cuda_vector<weight_type>));
    new (x) cuda_vector<int>();
    new (x_weight) cuda_vector<weight_type>();
    OUTs_Edges->push_back(x);
    OUTs_Edges_weight->push_back(x_weight);

    if (is_directed) {
        cuda_vector<int>* y;
        cuda_vector<weight_type>* y_weight;
        cudaMallocManaged(&y, sizeof(cuda_vector<int>));
        cudaMallocManaged(&y_weight, sizeof(cuda_vector<weight_type>));
        new (y) cuda_vector<int>();
        new (y_weight) cuda_vector<weight_type>();
        INs_Edges->push_back(y);
        INs_Edges_weight->push_back(y_weight);

        cuda_vector<int>* z;
        cuda_vector<weight_type>* z_weight;
        cudaMallocManaged(&z, sizeof(cuda_vector<int>));
        cudaMallocManaged(&z_weight, sizeof(cuda_vector<weight_type>));
        new (z) cuda_vector<int>();
        new (z_weight) cuda_vector<weight_type>();
        all_Edges->push_back(z);
        all_Edges_weight->push_back(z_weight);
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
GPU_adj<weight_type> to_GPU_adj(graph_structure<weight_type>& graph, bool is_directed = true) {
    GPU_adj<weight_type> gpu_adj;

    gpu_adj.is_directed = is_directed;

    gpu_adj.V = graph.size();
    gpu_adj.max_V = gpu_adj.V;

    gpu_adj.vertex_valid.resize(gpu_adj.max_V, true);
    
    cudaMallocManaged(&gpu_adj.OUTs_Edges, sizeof(cuda_vector<cuda_vector<int>*>));
    cudaMallocManaged(&gpu_adj.OUTs_Edges_weight, sizeof(cuda_vector<cuda_vector<weight_type>*>));
    cudaDeviceSynchronize();
    new (gpu_adj.OUTs_Edges) cuda_vector<cuda_vector<int>*>(gpu_adj.V);
    new (gpu_adj.OUTs_Edges_weight) cuda_vector<cuda_vector<weight_type>*>(gpu_adj.V);
    
    gpu_adj.OUTs_Edges->resize(gpu_adj.V);
    gpu_adj.OUTs_Edges_weight->resize(gpu_adj.V);
    
    for (int i = 0; i < gpu_adj.V; i++) {
        cudaMallocManaged(&((*gpu_adj.OUTs_Edges)[i]), sizeof(cuda_vector<int>));
        cudaMallocManaged(&((*gpu_adj.OUTs_Edges_weight)[i]), sizeof(cuda_vector<weight_type>));
        
        unsigned long long init_capacity = max(0ULL, (unsigned long long)(graph.OUTs[i].size() * RESERVATION_RATE));
        
        new ((*gpu_adj.OUTs_Edges)[i]) cuda_vector<int>(init_capacity);
        new ((*gpu_adj.OUTs_Edges_weight)[i]) cuda_vector<weight_type>(init_capacity);
    }
    
    if (is_directed) {

        cudaMallocManaged(&gpu_adj.INs_Edges, sizeof(cuda_vector<cuda_vector<int>*>));
        cudaMallocManaged(&gpu_adj.INs_Edges_weight, sizeof(cuda_vector<cuda_vector<weight_type>*>));
        cudaDeviceSynchronize();
        new (gpu_adj.INs_Edges) cuda_vector<cuda_vector<int>*>(gpu_adj.V);
        new (gpu_adj.INs_Edges_weight) cuda_vector<cuda_vector<weight_type>*>(gpu_adj.V);
        gpu_adj.INs_Edges->resize(gpu_adj.V);
        gpu_adj.INs_Edges_weight->resize(gpu_adj.V);
        for (int i = 0; i < gpu_adj.V; i++) {
            cudaMallocManaged(&((*gpu_adj.INs_Edges)[i]), sizeof(cuda_vector<int>));
            cudaMallocManaged(&((*gpu_adj.INs_Edges_weight)[i]), sizeof(cuda_vector<weight_type>));
            unsigned long long init_capacity = max(0ULL, (unsigned long long)(graph.INs[i].size() * RESERVATION_RATE));
            new ((*gpu_adj.INs_Edges)[i]) cuda_vector<int>(init_capacity);
            new ((*gpu_adj.INs_Edges_weight)[i]) cuda_vector<weight_type>(init_capacity);
        }

        for (int i = 0; i < gpu_adj.V; i++) {
            for (auto& xx : graph.INs[i])
                binary_insert((*gpu_adj.INs_Edges)[i], (*gpu_adj.INs_Edges_weight)[i], xx.first, xx.second);
        }

        cudaMallocManaged(&gpu_adj.all_Edges, sizeof(cuda_vector<cuda_vector<int>*>));
        cudaMallocManaged(&gpu_adj.all_Edges_weight, sizeof(cuda_vector<cuda_vector<weight_type>*>));
        new (gpu_adj.all_Edges) cuda_vector<cuda_vector<int>*>(gpu_adj.V);
        new (gpu_adj.all_Edges_weight) cuda_vector<cuda_vector<weight_type>*>(gpu_adj.V);
        gpu_adj.all_Edges->resize(gpu_adj.V);
        gpu_adj.all_Edges_weight->resize(gpu_adj.V);
        for (int i = 0; i < gpu_adj.V; i++) {
            cudaMallocManaged(&((*gpu_adj.all_Edges)[i]), sizeof(cuda_vector<int>));
            cudaMallocManaged(&((*gpu_adj.all_Edges_weight)[i]), sizeof(cuda_vector<weight_type>));
            unsigned long long init_capacity = max(0ULL, (unsigned long long)((graph.INs[i].size() + graph.OUTs[i].size()) * RESERVATION_RATE)); 
            new ((*gpu_adj.all_Edges)[i]) cuda_vector<int>(init_capacity);
            new ((*gpu_adj.all_Edges_weight)[i]) cuda_vector<weight_type>(init_capacity);
        }

        for (int i = 0; i < gpu_adj.V; i++) {
            for (auto& xx : graph.INs[i]) {
                (*gpu_adj.all_Edges)[i]->push_back(xx.first);
                (*gpu_adj.all_Edges_weight)[i]->push_back(xx.second);
            }
        }
        for (int i = 0; i < gpu_adj.V; i++) {
            for (auto& xx : graph.OUTs[i]) {
                (*gpu_adj.all_Edges)[i]->push_back(xx.first);
                (*gpu_adj.all_Edges_weight)[i]->push_back(xx.second);
            }
        }
    }

    for (int i = 0; i < gpu_adj.V; i++) {
        for (auto& xx : graph.OUTs[i])
            binary_insert((*gpu_adj.OUTs_Edges)[i], (*gpu_adj.OUTs_Edges_weight)[i], xx.first, xx.second);
    }

    return gpu_adj;
}

template <typename weight_type>
GPU_adj<weight_type>::~GPU_adj() {
    for (int i = 0; i < V; i++) {
        (*OUTs_Edges)[i]->~cuda_vector(), (*OUTs_Edges_weight)[i]->~cuda_vector();
        if (is_directed) {
            (*INs_Edges)[i]->~cuda_vector(), (*INs_Edges_weight)[i]->~cuda_vector();
            (*all_Edges)[i]->~cuda_vector(), (*all_Edges_weight)[i]->~cuda_vector();
        }
    }

    OUTs_Edges->~cuda_vector(), OUTs_Edges_weight->~cuda_vector();
    cudaFree(OUTs_Edges), cudaFree(OUTs_Edges_weight);
    if (is_directed) {
        INs_Edges->~cuda_vector(), INs_Edges_weight->~cuda_vector();
        all_Edges->~cuda_vector(), all_Edges_weight->~cuda_vector();
        cudaFree(INs_Edges), cudaFree(INs_Edges_weight);
        cudaFree(all_Edges), cudaFree(all_Edges_weight);
    }
}
