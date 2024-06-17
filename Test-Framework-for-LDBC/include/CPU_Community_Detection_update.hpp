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



    void cdlp_check(graph_structure<double>& graph, std::vector<long long int>& cpu_res,int & is_pass) {
    std::cout << "Checking CDLP results..." << std::endl;

    int size = cpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of CDLP results is not equal to the number of vertices!" << std::endl;
        return;
    }


    std::string base_line_file = "../results/" + graph.vertex_file;
    // remove the last two char

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-CDLP";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];
        
        if (cpu_res[v_id] != std::stoi(tokens[1])) {
            std::cout << "Baseline file and GPU CDLP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "GPU CDLP result: " << graph.vertex_id_to_str[cpu_res[v_id]] << " " << cpu_res[v_id] << std::endl;
            base_line.close();
            return;
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "CDLP results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}
*/
{   double start;
    double end;
    start = omp_get_wtime();
    int N = in_edge.size();
    std::vector<int> label(N);
    std::vector<int> new_label(N);

    for (int i = N - 1; i >= 0; i--)
    {
        label[i] = i;
    }
    
    for (int k = 0, total; k < iters; k++)
    {
        #pragma omp parallel for shared(in_edge,out_edge,label)
        for (int i = N - 1; i >= 0; i--)
        {   std::unordered_map<int, int> count;
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

        std::swap(new_label, label);
    }
        end = omp_get_wtime();
    std::vector<long long int>res(N);
    for (int i = 0; i < N; i++)
    {
        res[i] = strtoll(lab[label[i]].c_str(), NULL, 10);
    }
    
    printf("CPU CDLP cost time: %f s\n", (end - start));
    return res;
}