#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>
#include <ThreadPool.h>
#include <algorithm>

std::vector<double> PageRank(std::vector<std::vector<std::pair<int, double>>>& in_edge,
    std::vector<std::vector<std::pair<int, double>>>& out_edge, double damp, int iters) {

    int N = in_edge.size();

    std::vector<double> rank(N, 1 / N);
    std::vector<double> new_rank(N);

    double d = damp;
    double teleport = (1 - damp) / N;

    std::vector<int> sink;
    for (int i = 0; i < N; i++)
    {
        if (out_edge[i].size() == 0)
            sink.push_back(i);
    }

    for (int i = 0; i < iters; i++) {
        double sink_sum = 0;
        for (int i = 0; i < sink.size(); i++)
        {
            sink_sum += rank[sink[i]];
        }

        double x = sink_sum * d / N + teleport;

        /*P*/
        ThreadPool pool_dynamic(100);
        std::vector<std::future<int>> results_dynamic;
        for (int q = 0; q < 100; q++)
        {
            results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &rank, &out_edge, &new_rank, &x]
                {
                    int start = q * N / 100, end = std::min(N - 1, (q + 1) * N / 100);
                    for (int i = start; i <= end; i++) {
                        rank[i] /= out_edge[i].size();
                        new_rank[i] = x;
                    }

                    return 1; }));
        }
        for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic);

        for (int q = 0; q < 100; q++)
        {
            results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &in_edge, &rank, &new_rank, &d]
                {
                    int start = q * N / 100, end = std::min(N - 1, (q + 1) * N / 100);
                    for (int v = start; v <= end; v++) {
                        for (auto& y : in_edge[v]) {
                            new_rank[v] += d * rank[y.first];
                        }
                    }
                    return 1; }));
        }
        for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic);


        rank.swap(new_rank);
    }
    return rank;
}