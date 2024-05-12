#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include "./graph_structure/graph_structure.h"
using namespace std;
int ALPHA=0.85;
int ITERATION=10;

int GRAPHSIZE;
template <typename T>
class CSR
{
private:
    int edgeSize, graphSize;
    vector<double> outVec;
    vector<double> value;
    vector<int> val_col;
    vector<int> row_point;
    vector<int> row_size;

public:

    void makeCSR(graph_structure<T> &graph, int &GRAPHSIZE);
    inline vector<double> *multi_d_M_R(vector<double> *rankVector, double scaling);
    int get_edge_size()
    {
        return edgeSize;
    }
    int get_graph_size()
    {
        return graphSize;
    }
};

template <typename T>
void CSR::makeCSR(graph_structure<T> &graph, int &GRAPHSIZE)
{
    // cout << "makeCSR" << endl;
    int sumEdge = 0;
    GRAPHSIZE = graph.size();
    CSR_graph<double> ARRAY_graph;
    ARRAY_graph=graph.toCSR();
    row_point=ARRAY_graph.INs_Neighbor_start_pointers;
    for (int i = 0; i < GRAPHSIZE; i++)
    {
        // if (i == 0)
        // {
        //     row_point.push_back(0);
        // }
        // else
        // {
        //     row_point.push_back(row_point[i - 1] + graph.ADJs_T[i - 1].size());
        // }
        for (auto it : graph.INs[i])
        {
            value.push_back(1.0 / (graph.OUTs[it].size()));
            val_col.push_back(it);
        }
        sumEdge += graph.INs[i].size();
    }
    // row_point.push_back(row_point[GRAPHSIZE - 1] + graph.ADJs_T[GRAPHSIZE - 1].size());
    edgeSize = sumEdge;
    return;
}

inline vector<double> *CSR::multi_d_M_R(vector<double> *rankVector, double scaling)
{
    // cout << "multi_d_M_R" << endl;
    int sizeOfRankVec = rankVector->size();
    vector<double> *outVec = new vector<double>(sizeOfRankVec, 0);
    int colIndex = 0;

    for (int i = 0; i < sizeOfRankVec; i++)
    {
        for (int j = row_point[i]; j < row_point[i + 1]; j++)
        {
            colIndex = val_col[j];
            double valueInRow = value[j];
            (*outVec)[i] = (*outVec)[i] + (valueInRow * (*rankVector)[colIndex]);
            // cout << "i  j :" <<i<<"  " << j << "  colIndex : " << colIndex << "  valueInrRow : " << valueInRow << " *outVec[i] : " << (*outVec)[i] << "   rankVector[colIndex] : " << (*rankVector)[colIndex] << endl;
        }
        (*outVec)[i] *= scaling;
    }

    return outVec;
}

vector<double> *add_scaling(vector<double> *rankVector, double scaling)
{
    // cout << "add_scaling" << endl;
    vector<double> *outVec = new vector<double>(rankVector->size());
    for (int i = 0; i < rankVector->size(); i++)
    {
        (*outVec)[i] = (*rankVector)[i] + scaling;
    }
    return outVec;
}

vector<double> *add_vec(vector<double> *firVec, vector<double> *secVec)
{
    vector<double> *outVec = new vector<double>;
    int size = firVec->size();
    for (int i = 0; i < size; i++)
    {
        outVec->push_back((*firVec)[i] + (*secVec)[i]);
    }
    return outVec;
}

double vec_diff(vector<double> *oldRankVec, vector<double> *newRankVec)
{
    // cout << "vec_diff" << endl;
    int size = oldRankVec->size();
    double avgDiff = 0;
    double tempDiff = 0;
    for (int i = 0; i < size; i++)
    {
        tempDiff = abs((*newRankVec)[i] - (*oldRankVec)[i]);
        if (tempDiff > avgDiff)
        {
            avgDiff = tempDiff;
        }
    }
    return avgDiff;
}

vector<double> *Method(CSR *csr, vector<double> *rankVec, int &iteration)
{
    // cout << "Method" << endl;
    double diff = 1;

    int graphSize = csr->get_graph_size();
    int edgeSize = csr->get_edge_size();
    double d = ALPHA, d_ops = (1 - ALPHA) / graphSize;
    vector<double> *newRankVec = new vector<double>(graphSize);
    vector<double> *F = new vector<double>(graphSize);
    while (iteration<ITERATION)
    {
        F = csr->multi_d_M_R(rankVec, d);
        newRankVec = add_scaling(F, d_ops);
        // diff = vec_diff(rankVec, newRankVec);
        rankVec = newRankVec;
        iteration++;
        // cout << "diff :" << diff <<"  iteration : "<<iteration << endl;
    }

    // delete newRankVec;
    return rankVec;
}

int PageRank(graph_structure &graph)
{
    CSR *csr = new CSR;
    csr->makeCSR(graph,GRAPHSIZE);
    int size = csr->get_graph_size();
    double total = 0;
    ALPHA=graph.pr_damping;
    ITERATION=graph.cdlp_max_its;
    vector<double> *rank = new vector<double>(size, 1.0 / size);
    vector<double> *ans = new vector<double>(size);
    int iteration = 0;
    auto CPUstart = std::chrono::high_resolution_clock::now();
    ans = Method(csr, rank, iteration);
    auto CPUstop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(CPUstop - CPUstart);
    auto CPUtime = duration.count();
    total += CPUtime;

    cout << "CPU time : " << total << " ms" << endl;
    return 0;
}
