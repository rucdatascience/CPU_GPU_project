#pragma once
#include<iostream>
#include<vector>
#include<fstream>
#include<chrono>
#include<numeric>
#include<algorithm>
#include<limits>
#include"./graph_structure/graph_structure.h"
using namespace std;
static int GRAPHSIZE;
static int ITERAION=10;

vector<int> outs_row_ptr,ins_row_ptr, outs_neighbor,ins_neighbor; 
vector<int> labels,new_labels;

template <typename T>
void copy_init(graph_structure<T> &graph,int &GRAPHSIZE) {
    
    CSR_graph<int> ARRAY_graph;
    ARRAY_graph=graph.toCSR();
    GRAPHSIZE=ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;
    outs_row_ptr.resize(GRAPHSIZE + 1);
    outs_row_ptr=ARRAY_graph.OUTs_Neighbor_start_pointers;
    ins_row_ptr.resize(GRAPHSIZE + 1);
    ins_row_ptr=ARRAY_graph.INs_Neighbor_start_pointers;
    ins_neighbor=ARRAY_graph.INs_Edges;
    outs_neighbor=ARRAY_graph.OUTs_Edges;

    // cout << "CSR matrix created" << endl;
}

void init_label() {
    for (int i = 0; i < GRAPHSIZE; i++) {
        labels[i] = i;
        new_labels[i] = i;
    }
}

int findMostFrequentLabel(int outs_start, int outs_end,int ins_start,int ins_end) {
    vector<int> frequencyMap(GRAPHSIZE+1, 0);
    int mostFre = -1, mostFreLab = -1;
    for (int i = outs_start; i < outs_end; i++) {
        int neighbor = outs_neighbor[i];
        frequencyMap[labels_ptr[neighbor]]++;
        if (frequencyMap[labels_ptr[neighbor]] > mostFre) {
            mostFre = frequencyMap[labels_ptr[neighbor]];
            mostFreLab = labels_ptr[neighbor];
        }
        else if (frequencyMap[labels_ptr[neighbor]] == mostFre && labels_ptr[neighbor] < mostFreLab) {
            mostFreLab = labels_ptr[neighbor];
        }
    }
    for (int i = ins_start; i < ins_end; i++) {
        int neighbor = ins_neighbor[i];
        frequencyMap[labels_ptr[neighbor]]++;
        if (frequencyMap[labels_ptr[neighbor]] > mostFre) {
            mostFre = frequencyMap[labels_ptr[neighbor]];
            mostFreLab = labels_ptr[neighbor];
        }
        else if (frequencyMap[labels_ptr[neighbor]] == mostFre && labels_ptr[neighbor] < mostFreLab) {
            mostFreLab = labels_ptr[neighbor];
        }
    }

    return mostFreLab;
}

void labelPropagation() {
    int itertion=0;
    vector<int>* labels_ptr = &labels;
    vector<int>* new_labels_ptr = &new_labels;
    while (iteration<ITERAION) {
        for (int i = 0; i < GRAPHSIZE; ++i) {
            int outs_start = outs_row_ptr[i], outs_end = outs_row_ptr[i + 1];
            int ins_start = ins_row_ptr[i], ins_end = ins_row_ptr[i + 1];
            // if (start == end) continue; 

            int mostFrequentLabel = findMostFrequentLabel(outs_start, outs_end,ins_start,ins_end);
            
            if (labels_ptr[i] != mostFrequentLabel) {
                new_labels_ptr[i] = mostFrequentLabel;
            }else{
                new_labels_ptr[i] = labels_ptr[i];
            }
        }
        swap(labels_ptr, new_labels_ptr);
        iteration++;
    }
}

template <typename T>
int Community_Detection(graph_structure<T>& graph) {
    
    copy_init(graph, GRAPHSIZE);

    ITERAION=graph.cdlp_max_its;
    
    double CPUtime = 0;
    init_label();
    auto start = chrono::high_resolution_clock::now();
    labelPropagation();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end - start;
    CPUtime += duration.count();
    // cout << "CPU time: " << CPUtime / GRAPHSIZE << " ms" << endl;
}












