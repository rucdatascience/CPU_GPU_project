#include<iostream>
#include<vector>
#include<unordered_map>
#include<fstream>
#include<chrono>
#include<numeric>
#include<algorithm>
#include<limits>
#include <graph_structure/graph_structure.hpp>
using namespace std;
static int CPU_CD_GRAPHSIZE;
static int CPU_CD_ITERATION = 10;

static vector<int> outs_row_ptr, ins_row_ptr, outs_neighbor, ins_neighbor; 
static vector<int> labels, new_labels;

static vector<int>* labels_ptr = &labels;
static vector<int>* new_labels_ptr = &new_labels;

template <typename T>
void copy_init(graph_structure<T>& graph, int& CPU_CD_GRAPHSIZE) {
    CSR_graph<double> ARRAY_graph;
    ARRAY_graph = graph.toCSR();
    CPU_CD_GRAPHSIZE = ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;
    outs_row_ptr.resize(CPU_CD_GRAPHSIZE + 1);
    outs_row_ptr = ARRAY_graph.OUTs_Neighbor_start_pointers;
    ins_row_ptr.resize(CPU_CD_GRAPHSIZE + 1);
    ins_row_ptr = ARRAY_graph.INs_Neighbor_start_pointers;
    ins_neighbor = ARRAY_graph.INs_Edges;
    outs_neighbor = ARRAY_graph.OUTs_Edges;
    labels.resize(CPU_CD_GRAPHSIZE);
    new_labels.resize(CPU_CD_GRAPHSIZE);
}

void init_label() {
    for (int i = 0; i < CPU_CD_GRAPHSIZE; i++) {
        labels[i] = i;
        new_labels[i] = i;
    }
}

int findMostFrequentLabel(int ver,int outs_start, int outs_end, int ins_start, int ins_end) {
    unordered_map<int, int> frequencyMap;
    int mostFre = 1, mostFreLab = (*labels_ptr)[ver];
    int total=outs_end-outs_start+ins_end-ins_start+1;
    frequencyMap[(*labels_ptr)[ver]]++;
    for (int i = outs_start; i < outs_end; i++) {
        int neighbor = outs_neighbor[i];
        frequencyMap[(*labels_ptr)[neighbor]]++;
        if (frequencyMap[(*labels_ptr)[neighbor]] > mostFre) {
            mostFre = frequencyMap[(*labels_ptr)[neighbor]];
            mostFreLab = (*labels_ptr)[neighbor];
        }
        else if (frequencyMap[(*labels_ptr)[neighbor]] == mostFre && (*labels_ptr)[neighbor] < mostFreLab) {
            mostFreLab = (*labels_ptr)[neighbor];
        }
        if(mostFre > total/2) {
            return mostFreLab;
        }
    }

    for (int i = ins_start; i < ins_end; i++) {
        int neighbor = ins_neighbor[i];
        frequencyMap[(*labels_ptr)[neighbor]]++;
        if (frequencyMap[(*labels_ptr)[neighbor]] > mostFre) {
            mostFre = frequencyMap[(*labels_ptr)[neighbor]];
            mostFreLab = (*labels_ptr)[neighbor];
        }
        else if (frequencyMap[(*labels_ptr)[neighbor]] == mostFre && (*labels_ptr)[neighbor] < mostFreLab) {
            mostFreLab = (*labels_ptr)[neighbor];
        }
        if(mostFre > total/2) {
            return mostFreLab;
        }
    }

    return mostFreLab;
}

void labelPropagation(vector<int> &ans) {
    int iteration = 0;
    while (iteration < CPU_CD_ITERATION) {
        cout << "----iteration : " << iteration << " ----" << endl;
        for (int i = 0; i < CPU_CD_GRAPHSIZE; ++i) {
            int outs_start = outs_row_ptr[i], outs_end = outs_row_ptr[i + 1];
            int ins_start = ins_row_ptr[i], ins_end = ins_row_ptr[i + 1];
            int mostFrequentLabel = findMostFrequentLabel(i,outs_start, outs_end, ins_start, ins_end);

            if ((*labels_ptr)[i] != mostFrequentLabel) {
                (*new_labels_ptr)[i] = mostFrequentLabel;
            }
            else {
                (*new_labels_ptr)[i] = (*labels_ptr)[i];
            }
        }
        swap(labels_ptr, new_labels_ptr);
        iteration++;
    }
    ans=(*labels_ptr);
}

template <typename T>
int CPU_Community_Detection(graph_structure<T>& graph,vector<int> &ans) {
    copy_init(graph, CPU_CD_GRAPHSIZE);

    CPU_CD_ITERATION = graph.cdlp_max_its;

    init_label();

    labelPropagation(ans);

    return 0;
}

