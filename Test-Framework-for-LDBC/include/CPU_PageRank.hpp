#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <graph_structure/graph_structure.hpp>
using namespace std;

static int ALPHA = 0.85;
static int ITERATION = 10;

static int CPU_PR_GRAPHSIZE;

static vector<double> outVec;
static vector<double> value;
static vector<int> val_col_cpu;
static vector<int> row_point_cpu;
static vector<int> N_out_zero;
static vector<int> row_out_point_cpu;

vector<double> *multi_d_M_R(vector<double> *rankVector, double scaling)
{
    // cout << "multi_d_M_R" << endl;
    int sizeOfRankVec = rankVector->size();
    vector<double> *outVec = new vector<double>(sizeOfRankVec, 0);
    int colIndex = 0;

    for (int i = 0; i < sizeOfRankVec; i++)
    {
        for (int j = row_point_cpu[i]; j < row_point_cpu[i + 1]; j++)
        {
            colIndex = val_col_cpu[j];
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

    double d = ALPHA, d_ops = (1 - ALPHA) / CPU_PR_GRAPHSIZE;
    vector<double> *newRankVec = new vector<double>(CPU_PR_GRAPHSIZE);
    vector<double> *F = new vector<double>(CPU_PR_GRAPHSIZE);
    while (iteration < ITERATION)
    {
        double sink_sum=0;
        for (int i=0;i<N_out_zero.size();i++)
        {
            sink_sum+=rankVec->at(N_out_zero[i]);
        }

        F = multi_d_M_R(rankVec, d);
        newRankVec = add_scaling(F, d_ops+(ALPHA/CPU_PR_GRAPHSIZE)*sink_sum);
        // diff = vec_diff(rankVec, newRankVec);
        rankVec = newRankVec;
        iteration++;
        // cout << "diff :" << diff <<"  iteration : "<<iteration << endl;
    }

    // delete newRankVec;
    return rankVec;
}

void CPU_PageRank(graph_structure<double> &graph, vector<double> & result)
{
    CSR_graph<double> ARRAY_graph = graph.toCSR();
    CPU_PR_GRAPHSIZE = ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;

    row_point_cpu = ARRAY_graph.INs_Neighbor_start_pointers;
    row_out_point_cpu = ARRAY_graph.OUTs_Neighbor_start_pointers;
    for (int i = 0; i < CPU_PR_GRAPHSIZE; i++)
    {
        for (auto it : graph.INs[i])
        {
            value.push_back(1.0 / (graph.OUTs[it.first].size()));
            val_col_cpu.push_back(it.first);
        }
        if(row_out_point_cpu[i] == row_out_point_cpu[i + 1])
        {
            N_out_zero.push_back(i);
        }
       
    }

    double total = 0;
    ALPHA = graph.pr_damping;
    ITERATION = graph.pr_its;
    vector<double> *rank = new vector<double>(CPU_PR_GRAPHSIZE, 1.0 / CPU_PR_GRAPHSIZE);
    vector<double> *ans = new vector<double>(CPU_PR_GRAPHSIZE);
    
    int iteration = 0;
    auto CPUstart = std::chrono::high_resolution_clock::now();

    ans = Method( rank, iteration);

    auto CPUstop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(CPUstop - CPUstart);
    auto CPUtime = duration.count();
    total += CPUtime;

    result = *ans;
    // cout << "CPU time : " << total << " ms" << endl;
}
