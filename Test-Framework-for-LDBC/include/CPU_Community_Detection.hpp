#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <ThreadPool.h>

std::vector<long long int> CDLP(std::vector<std::vector<std::pair<int, double>>>& in_edge,
    std::vector<std::vector<std::pair<int, double>>>& out_edge, std::vector<long long int>& vertex_IDs_forCD, int iters)
    /*     call this function like:ans_cpu = CDLP(graph.INs, graph.OUTs,graph.vertex_id_to_str, graph.cdlp_max_its); */
{
    int N = in_edge.size();
    std::vector<long long int> label = vertex_IDs_forCD;
    std::vector<long long int> new_label(N);

    ThreadPool pool_dynamic(100);
    std::vector<std::future<int>> results_dynamic;

    for (int k = 0; k < iters; k++)
    {
        for (int q = 0; q < 100; q++)
        {
            results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &in_edge, &out_edge, &label, &new_label]
                {
                    int start = q * N / 100, end = std::min(N - 1, (q + 1) * N / 100);
                    for (int i = start; i <= end; i++) {

                        std::unordered_map<long long int, int> count;
                        for (auto& x: in_edge[i])
                        {
                            count[label[x.first]]++;
                        }
                        for (auto& x : out_edge[i])
                        {
                            count[label[x.first]]++;
                        }
                        int maxcount = 0;
                        long long int maxlabel = 0;
                        for (std::pair<long long int, int> p : count)
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

    return label;
}