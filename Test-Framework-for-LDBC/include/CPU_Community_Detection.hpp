#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>
#include <unordered_map>

std::vector<long long int> CDLP(std::vector<std::vector<std::pair<int, double>>> &in_edge,
                      std::vector<std::vector<std::pair<int, double>>> &out_edge, std::vector<long long int>& vertex_IDs_forCD, int iters)
/*     call this function like:ans_cpu = CDLP(graph.INs, graph.OUTs,graph.vertex_id_to_str, graph.cdlp_max_its); */
{
    int N = in_edge.size();
    std::vector<long long int> label = vertex_IDs_forCD;
    std::vector<long long int> new_label(N);
  
    for (int k = 0; k < iters; k++)
    {



        for (int i = N - 1; i >= 0; i--)
        {
            std::unordered_map<long long int, int> count;
            for (int j = in_edge[i].size() - 1; j >= 0; j--)
            {
                count[label[in_edge[i][j].first]]++;
            }
            for (int j = out_edge[i].size() - 1; j >= 0; j--)
            {
                count[label[out_edge[i][j].first]]++;
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




        std::swap(new_label, label);
    }

    return label;
}