#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <ThreadPool.h>
#include <numeric>

std::vector<std::string> CDLP(graph_structure<double>& graph, int iters)
/*     call this function like:ans_cpu = CDLP(graph.INs, graph.OUTs,graph.vertex_id_to_str, graph.cdlp_max_its); */
{
    auto& in_edges = graph.INs;
    auto& out_edges = graph.OUTs;

    int N = in_edges.size();
    std::vector<int> label(N);
    std::iota(std::begin(label), std::end(label), 0);
    std::vector<int> new_label(N);

    ThreadPool pool_dynamic(100);
    std::vector<std::future<int>> results_dynamic;

    for (int k = 0; k < iters; k++)
    {
        for (int q = 0; q < 100; q++)
        {
            results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &in_edges, &out_edges, &label, &new_label]
                {
                    int start = q * N / 100, end = std::min(N - 1, (q + 1) * N / 100);
                    for (int i = start; i <= end; i++) {

                        std::unordered_map<int, int> count;
                        for (auto& x : in_edges[i])
                        {
                            count[label[x.first]]++;
                        }
                        for (auto& x : out_edges[i])
                        {
                            count[label[x.first]]++;
                        }
                        int maxcount = 0;
                        int maxlabel = 0;
                        for (std::pair<int, int> p : count)
                        {
                            if (p.second > maxcount)
                            {
                                maxcount = p.second;
                                maxlabel = p.first;
                            }
                            else if (p.second == maxcount)
                            {
                                maxlabel = std::min(p.first, maxlabel);
                            }
                        }

                        new_label[i] = maxlabel;

                    }
                    return 1; }));
        }
        for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic);

        std::swap(new_label, label);
    }

    std::vector<std::string>res(N);
    for (int i = 0; i < N; i++)
    {
        res[i] = graph.vertex_id_to_str[label[i]];
    }

    return res;
}