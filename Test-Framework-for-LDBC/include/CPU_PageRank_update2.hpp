#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>
using namespace std;
double teleport, d;
int GRAPHSIZE;
std::vector<int> sink;


std::vector<double> PageRank(std::vector<std::vector<std::pair<int, double>>> in_edge, std::vector<std::vector<std::pair<int, double>>> out_edge, double damp, int iters)
{

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

    for (int i = 0; i < iters; i++)
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
            for (int j = 0; j < in_edge[i].size(); j++)
            {

                npr[i] += pr[in_edge[i][j].first];
            }
        }

        for (int i = 0; i < GRAPHSIZE; i++)
        {
            npr[i] *= d;
        }
        pr.swap(npr);

    }
    return pr;
}