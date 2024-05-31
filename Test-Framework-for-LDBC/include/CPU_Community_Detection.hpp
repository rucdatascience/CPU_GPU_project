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
static int CPU_CD_ITERATION = 10;//Iterate ten times

static vector<int> outs_row_ptr, ins_row_ptr, outs_neighbor, ins_neighbor; 
//The pointer indicates the starting and ending positions of the node's outgoing and incoming edges
static vector<int> labels, new_labels;//Static global variables

//Saving time by swapping old and new pointers
static vector<int>* labels_ptr = &labels;
static vector<int>* new_labels_ptr = &new_labels;

template <typename T>
void copy_init(graph_structure<T>& graph, int& CPU_CD_GRAPHSIZE) {
    CSR_graph<double> ARRAY_graph;
    ARRAY_graph = graph.toCSR();
    CPU_CD_GRAPHSIZE = ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;// get graphsize
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
        // Initially, each vertex v is assigned a unique label which matches its identifier.
        labels[i] = i;
        new_labels[i] = i;
    }
}

int findMostFrequentLabel(int ver,int outs_start, int outs_end, int ins_start, int ins_end) {
    unordered_map<int, int> frequencyMap;
    int mostFre = 1, mostFreLab = (*labels_ptr)[ver];//In case a vertex has no neighbors, it retains its current label.
    int total=outs_end-outs_start+ins_end-ins_start+1;//The number of neighbors of a vertex
    frequencyMap[(*labels_ptr)[ver]]++;
    for (int i = outs_start; i < outs_end; i++) {
        int neighbor = outs_neighbor[i];//get neighbor
        frequencyMap[(*labels_ptr)[neighbor]]++;//Increase the number of labels corresponding to neighbors
        if (frequencyMap[(*labels_ptr)[neighbor]] > mostFre) {//update the most frequently label
            mostFre = frequencyMap[(*labels_ptr)[neighbor]];
            mostFreLab = (*labels_ptr)[neighbor];
        }
        else if (frequencyMap[(*labels_ptr)[neighbor]] == mostFre && (*labels_ptr)[neighbor] < mostFreLab) {
            mostFreLab = (*labels_ptr)[neighbor];
            //In case there are multiple labels with the maximum frequency, the smallest label is chosen. 
        }
        if(mostFre > total/2) {
           // If a label appears more than half the number of neighbors, then it must be the label that appears the most frequently
            return mostFreLab;
        }
    }

    for (int i = ins_start; i < ins_end; i++) {
        int neighbor = ins_neighbor[i];//get neighbor
        frequencyMap[(*labels_ptr)[neighbor]]++;//Increase the number of labels corresponding to neighbors
        if (frequencyMap[(*labels_ptr)[neighbor]] > mostFre) {//update the most frequently label
            mostFre = frequencyMap[(*labels_ptr)[neighbor]];
            mostFreLab = (*labels_ptr)[neighbor];
        }
        else if (frequencyMap[(*labels_ptr)[neighbor]] == mostFre && (*labels_ptr)[neighbor] < mostFreLab) {
            mostFreLab = (*labels_ptr)[neighbor];
             //In case there are multiple labels with the maximum frequency, the smallest label is chosen.
        }
        if(mostFre > total/2) {
            // If a label appears more than half the number of neighbors, then it must be the label that appears the most frequently
            return mostFreLab;
        }
    }

    return mostFreLab;
}
/* In iteration i, each vertex v determines the frequency of the labels of its incoming and outgoing neighbors
and selects the label which is most common.  */
void labelPropagation(vector<int> &ans) {
    
    int iteration = 0;
    while (iteration < CPU_CD_ITERATION) {
        cout << "----iteration : " << iteration << " ----" << endl;
        for (int i = 0; i < CPU_CD_GRAPHSIZE; ++i) {
            int outs_start = outs_row_ptr[i], outs_end = outs_row_ptr[i + 1];
            int ins_start = ins_row_ptr[i], ins_end = ins_row_ptr[i + 1];
            //Obtain the range of in and out edges
            int mostFrequentLabel = findMostFrequentLabel(i,outs_start, outs_end, ins_start, ins_end);
            
            if ((*labels_ptr)[i] != mostFrequentLabel) {
                (*new_labels_ptr)[i] = mostFrequentLabel;
            }
            else {
                (*new_labels_ptr)[i] = (*labels_ptr)[i];
            }
           /*  If it is found that the most frequent label obtained in this 
            round is different from the original most frequent label,
             it should be modified to the new most frequent label. 
             Otherwise, the original most frequent label should be kept */
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

