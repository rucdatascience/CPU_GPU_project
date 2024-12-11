#pragma once
#include <CPU_adj_list/CPU_adj_list.hpp>
#include <vector>
#include <iostream>
#include <CPU_adj_list/ThreadPool.h>
#include <algorithm>

// PageRank Algorithm
// call this function like: ans_cpu = CPU_PR(graph, damp, graph.cdlp_max_its);
// used to show the relevance and importance of vertices in the graph
// return the pagerank of each vertex based on the graph, damping factor and number of iterations.
std::vector<double> PageRank (std::vector<std::vector<std::pair<int, double>>>& in_edge,
    std::vector<std::vector<std::pair<int, double>>>& out_edge, double damp, int iters) {

    int N = in_edge.size(); // number of vertices in the graph

    std::vector<double> rank(N, 1 / (double)N); // The initial pagerank of each vertex is 1/|V|
    std::vector<double> new_rank(N); // temporarily stores the updated pagerank

    double teleport = (1 - damp) / N; // teleport mechanism

    std::vector<int> sink; // the set of sink vertices
    for (int i = 0; i < N; i++)
    {
        if (out_edge[i].size() == 0)
            sink.push_back(i); // record the sink vertices
    }

    for (int i = 0; i < iters; i++) {
        double sink_sum = 0;
        for (int idx : sink)
            sink_sum += rank[idx];

        double common_add = (sink_sum * damp / N) + teleport;
        ThreadPool pool_dynamic(100);

        std::vector<std::future<void>> futures;
        for (int q = 0; q < 100; q++) {
            futures.emplace_back(pool_dynamic.enqueue([q, N, &rank, &out_edge, &new_rank, &common_add, &damp, &in_edge] {
                int start = (long long)q * N / 100,
                    end = std::min((long long)N - 1, (long long)(q + 1) * N / 100);
                for (int v = start; v <= end; v++) {
                    double sum_contrib = 0;
                    for (const auto& in : in_edge[v])
                        sum_contrib += rank[in.first] / out_edge[in.first].size();
                    new_rank[v] = common_add + damp * sum_contrib;
                }
            }));
        }
        for (auto& f : futures)
            f.get();

        rank.swap(new_rank);
    }

    return rank; // return the pagerank of each vertex
}

// PageRank Algorithm
// return the pagerank of each vertex based on the graph, damping factor and number of iterations.
// the type of the vertex and pagerank are string
std::vector<std::pair<std::string, double>> CPU_PR (graph_structure<double>& graph, int iterations, double damping) {
    std::vector<double> prValueVec = PageRank(graph.INs, graph.OUTs, damping, iterations); // get the pagerank in double type
    return graph.res_trans_id_val(prValueVec);
}