#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>
#include <mutex>

using namespace std;
double teleport, d;
int GRAPHSIZE;
std::vector<int> sink;

std::vector<double> PageRank(std::vector<std::vector<std::pair<int, double>>>& in_edge,
    std::vector<std::vector<std::pair<int, double>>>& out_edge, double damp, int iters)
{
    double start;
    double end;

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
        for (int i = 0; i < sink.size(); i++)
        {
            red += pr[sink[i]];
        }
        for (int i = 0; i < GRAPHSIZE; i++)
        {
            pr[i] /= out_edge[i].size();
        }


        red = red * d / GRAPHSIZE;
        std::fill(npr.begin(), npr.end(), teleport + red);
        for (int i = 0; i < GRAPHSIZE; i++)
        {
            double temp = npr[i];

            for (int j = 0; j < in_edge[i].size(); j++)
            {
                temp += pr[in_edge[i][j].first];
            }
            npr[i] = temp * d;
        }



        pr.swap(npr);
    }

    printf("CPU PageRank cost time: %f ms\n", (end - start) * 1000);
    return pr;
}