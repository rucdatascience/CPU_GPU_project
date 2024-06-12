#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>

std::vector<double> PageRank(std::vector<std::vector<std::pair<int, double>>>& in_edge,
    std::vector<std::vector<std::pair<int, double>>>& out_edge, double damp, int iters)
{

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

    for (int i = 0; i < iters; i++)
    {
        double sink_sum = 0;
        for (int i = 0; i < sink.size(); i++)
        {
            sink_sum += rank[sink[i]];
        }
        for (int i = 0; i < N; i++)
        {
            rank[i] /= out_edge[i].size();
        }

        sink_sum = sink_sum * d / N;
        std::fill(new_rank.begin(), new_rank.end(), teleport + sink_sum);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < in_edge[i].size(); j++)
            {
                new_rank[i] += rank[in_edge[i][j].first];
            }
        }

        for (int i = 0; i < N; i++)
        {
            new_rank[i] *= d;
        }
        rank.swap(new_rank);

    }
    return rank;
}