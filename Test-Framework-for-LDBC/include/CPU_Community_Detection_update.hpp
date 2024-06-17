#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <omp.h>
std::vector<long long int> CDLP(std::vector<std::vector<std::pair<int, double>>> &in_edge,
                      std::vector<std::vector<std::pair<int, double>>> &out_edge, std::vector<std::string> lab, int iters)
/*     call this function like:ans_cpu = CDLP(graph.INs, graph.OUTs,graph.vertex_id_to_str, graph.cdlp_max_its); 
/* to run this file,you need add below code to Cmakelists:
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
you must not use clock() function in test.cpp!

if (graph.sup_cdlp) {
        int cdlp_pass = 0;
        std::vector<long long int> ans_cpu;
        std::vector<int> ans_gpu;


        ans_cpu = CDLP(graph.INs, graph.OUTs,graph.vertex_id_to_str, graph.cdlp_max_its);


        cdlp_check(graph, ans_cpu, cdlp_pass);
        
        string cdlp_pass_label = "No";
        if(cdlp_pass != 0) cdlp_pass_label = "Yes";
        umap_all_res.emplace("cdlp_pass_label", cdlp_pass_label);        
    }
*/
{   double start;
    double end;
    start = omp_get_wtime();
    int N = in_edge.size();
    std::vector<long long int> label(N);
    std::vector<long long int> new_label(N);

    for (int i = N - 1; i >= 0; i--)
    {
        label[i] =strtoll(lab[i].c_str(), NULL, 10);
    }
    
    for (int k = 0, total; k < iters; k++)
    {
        #pragma omp parallel for shared(in_edge,out_edge,label)
        for (int i = N - 1; i >= 0; i--)
        {   std::unordered_map<long long int, int> count;
            total = (in_edge[i].size() + out_edge[i].size()) / 2;
            count.clear();
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
        end = omp_get_wtime();

    printf("CPU CDLP cost time: %f s\n", (end - start) * 1000);
    return label;
}