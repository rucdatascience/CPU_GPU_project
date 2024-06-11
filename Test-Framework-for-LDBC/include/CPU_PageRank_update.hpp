#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>
#include <mutex>
#include <omp.h>
/* to run this file,you need add below code to Cmakelists:
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
you must not use clock() function in test.cpp!

if (graph.sup_pr) {
        int pr_pass = 0;
        
        vector<double> cpu_pr_result, gpu_pr_result;
        cpu_pr_result = PageRank(graph.INs,graph.OUTs,graph.pr_damping,graph.pr_its);
       
        pr_checker_cpu(graph, cpu_pr_result, pr_pass);
   
    }

 */
using namespace std;
double teleport, d;
int GRAPHSIZE;
std::vector<int> sink;

std::vector<double> PageRank(std::vector<std::vector<std::pair<int, double>>> &in_edge, std::vector<std::vector<std::pair<int, double>>> &out_edge, double damp, int iters)
{
    double start; 
double end; 
start = omp_get_wtime(); 
    
    GRAPHSIZE = in_edge.size();

    std::vector<double> pr(GRAPHSIZE, 1 / GRAPHSIZE);
    std::vector<double> npr(GRAPHSIZE);
    d = damp;
    teleport = (1 - damp) / GRAPHSIZE;
    for (int i = 0; i < GRAPHSIZE; i++)
    {
        if (out_edge[i].size() == 0)
            sink.push_back(i);
    }
    
    for (int k = 0; k < iters; k++)
    {
        double red = 0;
            #pragma omp parallel for shared(pr,sink) reduction(+ : red)
            for (int i = 0; i < sink.size(); i++)
            {
                red += pr[sink[i]];
            }
            #pragma omp parallel for shared(pr,out_edge)
            for (int i = 0; i < GRAPHSIZE; i++)
            {
                pr[i] /= out_edge[i].size();
            }
        

        red = red * d / GRAPHSIZE;
        std::fill(npr.begin(), npr.end(), teleport + red);
        #pragma omp parallel for shared(pr,in_edge,npr)
        for (int i = 0; i < GRAPHSIZE; i++)
        {   
            double temp = npr[i];
            
            for (int j = 0; j < in_edge[i].size(); j++)
            {
                temp += pr[in_edge[i][j].first];
            }
            npr[i] = temp*d;
        }

        

        pr.swap(npr);
    }
end = omp_get_wtime(); 

printf("CPU PageRank cost time: %f ms\n", (end - start)*1000); 
    return pr;
}
