#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include "./graph_structure/graph_structure.h"
using namespace std;
int ALPHA = 0.85;
int ITERATION = 10;

int GRAPHSIZE;

vector<double> outVec;
vector<double> value;
vector<int> val_col;
vector<int> row_point;

vector<double> *multi_d_M_R(vector<double> *rankVector, double scaling);


vector<double> *CSR::multi_d_M_R(vector<double> *rankVector, double scaling)
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

// double vec_diff(vector<double> *oldRankVec, vector<double> *newRankVec)
// {
//     // cout << "vec_diff" << endl;
//     int size = oldRankVec->size();
//     double avgDiff = 0;
//     double tempDiff = 0;
//     for (int i = 0; i < size; i++)
//     {
//         tempDiff = abs((*newRankVec)[i] - (*oldRankVec)[i]);
//         if (tempDiff > avgDiff)
//         {
//             avgDiff = tempDiff;
//         }
//     }
//     return avgDiff;
// }

vector<double> *Method( vector<double> *rankVec, int &iteration)
{
    // cout << "Method" << endl;
    double diff = 1;


    double d = ALPHA, d_ops = (1 - ALPHA) / GRAPHSIZE;
    vector<double> *newRankVec = new vector<double>(GRAPHSIZE);
    vector<double> *F = new vector<double>(GRAPHSIZE);
    while (iteration < ITERATION)
    {
        F = multi_d_M_R(rankVec, d);
        newRankVec = add_scaling(F, d_ops);
        // diff = vec_diff(rankVec, newRankVec);
        rankVec = newRankVec;
        iteration++;
        // cout << "diff :" << diff <<"  iteration : "<<iteration << endl;
    }

    // delete newRankVec;
    return rankVec;
}

void PageRank(graph_structure &graph)
{
    CSR_graph<double> ARRAY_graph = graph.toCSR();
    GRAPHSIZE = ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;

    row_point = ARRAY_graph.INs_Neighbor_start_pointers;
    
    for (int i = 0; i < GRAPHSIZE; i++)
    {
        for (auto it : graph.INs[i])
        {
            value.push_back(1.0 / (graph.OUTs[it.first].size()));
            val_col.push_back(it.first);
        }
       
    }

    double total = 0;
    ALPHA = graph.pr_damping;
    ITERATION = graph.cdlp_max_its;
    vector<double> *rank = new vector<double>(GRAPHSIZE, 1.0 / GRAPHSIZE);
    vector<double> *ans = new vector<double>(GRAPHSIZE);
    
    int iteration = 0;
    auto CPUstart = std::chrono::high_resolution_clock::now();

    ans = Method( rank, iteration);

    auto CPUstop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(CPUstop - CPUstart);
    auto CPUtime = duration.count();
    total += CPUtime;

    // cout << "CPU time : " << total << " ms" << endl;
}
