#include<iostream>
#include<vector>
#include<unordered_map>
#include<algorithm>
#include<random>
#include <time.h>
#include <graph_structure/graph_structure.hpp>
#include <checker.hpp>
#include <checker_cpu.hpp>
using namespace std;

vector<int> community_detection_CDP(graph_structure<double> &G, int max_iterations) {
    vector<int> labels(G.size());
    vector<int> new_labels(G.size());

    // Initialize labels
    for (int v = 0; v < G.size(); ++v) {
        labels[v] = v;
    }

    // Label propagation iterations
    for (int i = 0; i < max_iterations; ++i) {
        for (int v = 0; v < G.size(); ++v) {
            unordered_map<int, int> C_histogram;

            // Count labels in INs
            for (auto &edge : G.INs[v]) {
                int u = edge.first;
                C_histogram[labels[u]]++;
            }

            // Count labels in OUTs
            for (auto &edge : G.OUTs[v]) {
                int u = edge.first;
                C_histogram[labels[u]]++;
            }

            // Find label with maximum frequency
            int freq = -1;
            int candidates = labels[v];
            for (auto &entry : C_histogram) {
                // if (entry.second > freq 
                // || (entry.second == freq && entry.first < candidates)) {
                //     freq = entry.second;
                //     candidates = entry.first;
                // }

                if(entry.second > freq){
                    freq = entry.second;
                }
            }

            for(auto & entry : C_histogram){
                if(entry.second == freq && entry.first < candidates){
                    candidates = entry.first;
                }
            }

            new_labels[v] = candidates;
        }

        // Update labels for next iteration
        labels = new_labels;
    }

    return labels;
}


void test(graph_structure<double> graph){
        float elapsedTime = 0;

        clock_t start = clock(), end = clock();
        int cdlp_pass = 0;
        
        vector<int> cdlp_result = community_detection_CDP(graph, graph.cdlp_max_its);
        end = clock();
        double time_cpu_cdlp = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("CPU_cdlp cost time: %f ms\n", time_cpu_cdlp);

        cdlp_check_cpu(graph, cdlp_result, cdlp_pass);
}

// int main(){

//     graph_structure<double> graph;
//     std::string config_file = "../data/datagen-7_5-fb.properties";//quick test

//     graph.read_config(config_file);
//     graph.load_LDBC();
//     int max_iter = graph.cdlp_max_its;

//     // unordered_map<int, int> labels = detection_community(graph, max_iter);

//     // cout << "Vertex Communities:" << endl;
//     // for (const auto& pair : labels) {
//     //     cout << "Vertex " << pair.first << ": Community " << pair.second << endl;
//     // }

//     test(graph);

//     return 0;
// }