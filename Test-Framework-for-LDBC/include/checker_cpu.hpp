#pragma once

#include <graph_structure/graph_structure.hpp>
#include <algorithm>
#include <cmath>
#include <checker.hpp>
#include <iostream>
#include "checker.hpp"
using namespace std;
void saveResult(string path, vector<int> & res);

void bfs_checker_cpu(graph_structure<double>& graph, std::vector<int>& cpu_res, int & cpu_bfs_pass){
    std::cout << "Checking CPU_BFS results..." << std::endl;
    
    int size = cpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of BFS results is not equal to the number of vertices!" << std::endl;
        return;
    }

    std::string base_line_file = "../results/" + graph.vertex_file;
    // remove the last two char

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-BFS";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }
        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];
        if (cpu_res[v_id] != std::stol(tokens[1])) {
            std::cout << "Baseline file and CPU BFS results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "CPU BFS result: " << graph.vertex_id_to_str[cpu_res[v_id]] << " " << cpu_res[v_id] << std::endl;
            base_line.close();
            return;
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "CPU_BFS results are correct!" << std::endl;
    cpu_bfs_pass = 1;
    base_line.close();

}

void wcc_checker_cpu(graph_structure<double>& graph, std::vector<std::vector<int>>& cpu_res, int & cpu_wcc_pass){
    std::cout << "Checking CPU_WCC results..." << std::endl;
    
    int size = cpu_res.size();

    for (auto &v : cpu_res) {
        if (!v.size()) {
            std::cout << "One of CPU WCC results is empty!" << std::endl;
            return;
        }
        std::sort(v.begin(), v.end());
    }

    std::sort(cpu_res.begin(), cpu_res.end(), compare);

    std::string base_line_file = "../results/" + graph.vertex_file;

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-WCC";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    std::vector<std::vector<int>> base_res;
    std::vector<int> component;

    component.resize(graph.V, 0);

    std::string line;

    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        //store baseline file per row value to component
        component[graph.vertex_str_to_id[tokens[0]]] = graph.vertex_str_to_id[tokens[1]];
    }

    std::vector<std::vector<int>> componentLists(graph.V);

    for (int i = 0; i < graph.V; i++) {
        componentLists[component[i]].push_back(i);
    }

    for (int i = 0; i < graph.V; i++) {
		if (componentLists[i].size() > 0)
			base_res.push_back(componentLists[i]);
	}

     for (auto &v : base_res) {
        if (!v.size()) {
            std::cout << "One of baseline WCC results is empty!" << std::endl;
            base_line.close();
            return;
        }
        std::sort(v.begin(), v.end());
    }

    std::sort(base_res.begin(), base_res.end(), compare);

    for (int i = 0; i < size; i++) {
        if (base_res[i].size() != cpu_res[i].size()) {
            std::cout << "Baseline file and GPU WCC results are not the same!" << std::endl;
            std::cout << "Baseline file: " << graph.vertex_id_to_str[i] << " " << base_res[i][0] << std::endl;
            std::cout << "CPU_WCC result: " << graph.vertex_id_to_str[cpu_res[i][0]] << " " << cpu_res[i][0] << std::endl;
            base_line.close();
            return;
        }
        for (int j = 0; j < base_res[i].size(); j++) {
            if (base_res[i][j] != cpu_res[i][j]) {
                std::cout << "Baseline file and GPU WCC results are not the same!" << std::endl;
                std::cout << "Difference at: " << graph.vertex_id_to_str[base_res[i][j]] << " " << graph.vertex_id_to_str[cpu_res[i][j]] << std::endl;
                base_line.close();
                return;
            }
        }
    }

    std::cout << "CPU_WCC results are correct!" << std::endl;
    cpu_wcc_pass = 1;
    base_line.close();

}

void pr_checker_cpu(graph_structure<double>& graph, std::vector<double>& cpu_res, int & cpu_pr_pass){
    std::cout << "Checking CPU PageRank results..." << std::endl;
    int size = cpu_res.size();
    cout<<"cpu result size:"<<size<<endl;

    if (size != graph.V) {
        std::cout << "Size of CPU PageRank results is not equal to the number of vertices!" << std::endl;
        return;
    }

    std::string base_line_file = "../results/" + graph.vertex_file;
    // remove the last two char

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-PR";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];

        if (fabs(cpu_res[v_id] - std::stod(tokens[1])) > 1e-4) {
            std::cout << "Baseline file and GPU PageRank results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "GPU PageRank result: " << graph.vertex_id_to_str[v_id] << " " << cpu_res[v_id] << std::endl;
            base_line.close();
            return;
        }
        id++;
    }

    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "CPU PageRank results are correct!" << std::endl;
    cpu_pr_pass = 1;
    base_line.close();

}

void cdlp_check_cpu(graph_structure<double>& graph, std::vector<int>& cpu_res, int & cpu_cd_pass){
    std::cout << "Checking CPU CDLP results..." << std::endl;
    int size = cpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of CDLP results is not equal to the number of vertices!" << std::endl;
        return;
    }

    std::string base_line_file = "../results/" + graph.vertex_file;
    // remove the last two char

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-CDLP";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];
        
        if (cpu_res[v_id] != std::stol(tokens[1])) {
            std::cout << "Baseline file and CPU CDLP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "CPU CDLP result: " << graph.vertex_id_to_str[cpu_res[v_id]] << " " << cpu_res[v_id] << std::endl;
            base_line.close();
            return;
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "CPU CDLP results are correct!" << std::endl;
    cpu_cd_pass = 1;
    base_line.close();

}

void cdlp_cross_validate(graph_structure<double> & graph, vector<int> & ht_cdlp, 
        vector<int> & gpu_cdlp, vector<int> & lp_cdlp){

    std::cout << "+++++++++++Checking cdlp_cross_validate CDLP results..." << std::endl;
    int size = ht_cdlp.size();

    if (size != graph.V) {
        std::cout << "Size of cdlp_cross_validate CDLP results is not equal to the number of vertices!" << std::endl;
        return;
    }

    for(int i = 0; i < size; ++i){
        if(ht_cdlp[i] != gpu_cdlp[i]){
            cout<<"ht cpu and gpu result is not same!"<<endl;
            return ;
        }
    }
    cout <<"ht cpu and gpu result have pass!"<<endl;

    int size2 = lp_cdlp.size();
    
    if(size != size2){
        cout <<"ht and lp have different size!"<<endl;
        return ;
    }

    string path = "../data/ht_cdlp.txt";
    string path2 = "../data/lp_cdlp.txt";

    saveResult(path, ht_cdlp);
    cout<<"########ht_cdlp result have stored!"<<endl;

    saveResult(path2, lp_cdlp);
    cout<<"$$$$$$$$lp_cdlp result have stored!"<<endl;
    
    int count = 0;
    for(int i = 0; i < size2; ++i){
        if(ht_cdlp[i] != lp_cdlp[i]){
            // cout<<"ht_cdlp_cpu is not same lp_cdlp_cpu!"<<endl;
            // cout<<"the difference is:"<<i+1<<" , ht_cdlp:"<<ht_cdlp[i]<<" , lp_cdlp:"<<lp_cdlp[i]<<endl;
            count++;
        }
    }

    cout <<"ht cdlp with lp cdlp different number is="<<count<<endl;

    // readBaseline(graph, ht_cdlp);

}

void saveResult(string path, vector<int> & res){

    std::ofstream fileOutput(path);

    if(!fileOutput.is_open()){
        std::cerr<<"the file can't open or exist!"<<endl;
        return;
    }

    for(auto & iter : res){
        fileOutput<<iter<<endl;
    }

    fileOutput.close();
    cout<<"the result have store finished!"<<endl;

}