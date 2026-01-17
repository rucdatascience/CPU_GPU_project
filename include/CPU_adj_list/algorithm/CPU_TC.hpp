#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>

#include <CPU_adj_list/CPU_adj_list.hpp>

std::vector<std::vector<int>> build_directed_adj(const std::vector<std::vector<std::pair<int, double>>> &undir_adj) {
    int N = undir_adj.size();
    
    std::vector<int> deg(N);
    for (int u = 0; u < N; ++u) {
        deg[u] = static_cast<int>(undir_adj[u].size());
    }
    
    std::vector<std::vector<int>> dir_adj(N);
    for (int u = 0; u < N; ++u) {
        for (const auto &edge : undir_adj[u]) {
            int v = edge.first;
            if (deg[u] < deg[v] || (deg[u] == deg[v] && u < v)) {
                dir_adj[u].push_back(v);
            }
        }
        std::sort(dir_adj[u].begin(), dir_adj[u].end());
    }

    return dir_adj;
}

std::vector<int> Tri_Counting(const std::vector<std::vector<std::pair<int, double>>> &input_graph) {
    int N = input_graph.size();
    std::vector<int> tri_count(N, 0);

    for (int u = 0; u < N; ++ u) {
        const auto &Nu = input_graph[u];
        for (const auto &edge : Nu) {
            int v = edge.first;
            const auto &Nv = input_graph[v];
            size_t i = 0, j = 0;
            while (i < Nu.size() && j < Nv.size()) {
                int w_u = Nu[i].first;
                int w_v = Nv[j].first;
                if (w_u == w_v) {
                    tri_count[u] ++;
                    tri_count[v] ++;
                    tri_count[w_u] ++;
                    ++ i;
                    ++ j;
                } else if (w_u < w_v) {
                    ++ i;
                } else {
                    ++ j;
                }
            }
        }
    }

    return tri_count;
}

std::vector<std::pair<std::string, int>> CPU_TC(graph_structure<double> &graph) {
    std::vector<int> tri_counts = Tri_Counting(graph.OUTs);
    return graph.res_trans_id_val(tri_counts);
}