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
int GRAPHSIZE;


vector<int> row_ptr, col_indices; 
vector<int> labels;

void make_csr(graph_structure &graph) {
    GRAPHSIZE = graph.ADJs.size();
    // cout<<GRAPHSIZE<<endl;
    row_ptr.resize(GRAPHSIZE + 1);
    row_ptr[0] = 0;
    for (int i = 0; i < GRAPHSIZE; i++) {
        for (int j :graph.ADJs[i]) {
            col_indices.push_back(j.first);
        }
        row_ptr[i + 1] = row_ptr[i] + graph.ADJs[i].size();
    }

    cout << "CSR matrix created" << endl;
}

void init_label() {
    for (int i = 0; i < GRAPHSIZE; i++) {
        labels[i] = i;
    }
}

int findMostFrequentLabel(int start, int end) {
    vector<int> frequencyMap(GRAPHSIZE, 0);
    int mostFre = -1, mostFreLab = -1;
    for (int i = start; i < end; i++) {
        int neighbor = col_indices[i];
        frequencyMap[labels[neighbor]]++;
        if (frequencyMap[labels[neighbor]] > mostFre) {
            mostFre = frequencyMap[labels[neighbor]];
            mostFreLab = labels[neighbor];
        }
        else if (frequencyMap[labels[neighbor]] == mostFre && labels[neighbor] < mostFreLab) {
            mostFreLab = labels[neighbor];
        }
    }

    return mostFreLab;
}

void labelPropagation() {
    bool keepUpdating = true;
    while (keepUpdating) {
        keepUpdating = false;
        for (int i = 0; i < GRAPHSIZE; ++i) {
            int start = row_ptr[i], end = row_ptr[i + 1];
            if (start == end) continue; 

            int mostFrequentLabel = findMostFrequentLabel(start, end);
            if (labels[i] != mostFrequentLabel) {
                labels[i] = mostFrequentLabel;
                keepUpdating = true;
            }
        }
    }
}

int Community_Detection(graph_structure& graph) {
    make_csr(graph, GRAPHSIZE);
    double CPUtime = 0;
    init_label();
    auto start = chrono::high_resolution_clock::now();
    labelPropagation();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end - start;
    CPUtime += duration.count();
    cout << "CPU time: " << CPUtime / GRAPHSIZE << " ms" << endl;
}












