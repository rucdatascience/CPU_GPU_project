#pragma once

#include <graph_structure/graph_structure.hpp>
#include <algorithm>
#include <cmath>
#include <checker.hpp>
#include "checker_cpu.hpp"

void bfs_checker_gpu(graph_structure<double>& graph, std::vector<unsigned long long int>& gpu_res, int & gpu_bfs_pass){
    std::cout << "Checking GPU_BFS results..." << std::endl;

    // string path = "/home/liupeng/CPU_GPU_project/Test-Framework-for-LDBC/results/gpu_bfs_patents.txt";
    // saveResult(path, gpu_res);
    
    int size = gpu_res.size();

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
        if (gpu_res[v_id] != std::stol(tokens[1])) {
            std::cout << "Baseline file and GPU BFS results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "GPU BFS result: " << graph.vertex_id_to_str[gpu_res[v_id]] << " " << gpu_res[v_id] << std::endl;
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

    std::cout << "GPU_BFS results are correct!" << std::endl;
    gpu_bfs_pass = 1;
    base_line.close();

}

void pr_checker_gpu(graph_structure<double>& graph, std::vector<double>& gpu_res, int & gpu_pr_pass){
    // std::cout << "Checking GPU PageRank results..." << std::endl;

    // string path = "/home/liupeng/CPU_GPU_project/Test-Framework-for-LDBC/results/gpu_pr_patents.txt";
    // saveResult(path, gpu_res);

    int size = gpu_res.size();
    cout<<"cpu result size:"<<size<<endl;

    if (size != graph.V) {
        std::cout << "Size of GPU PageRank results is not equal to the number of vertices!" << std::endl;
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

        if (fabs(gpu_res[v_id] - std::stod(tokens[1])) > 1e-4) {
            std::cout << "Baseline file and GPU PageRank results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "GPU PageRank result: " << graph.vertex_id_to_str[v_id] << " " << gpu_res[v_id] << std::endl;
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

    std::cout << "GPU PageRank results are correct!" << std::endl;
    gpu_pr_pass = 1;
    base_line.close();

}

void wcc_checker_gpu(graph_structure<double>& graph, std::vector<std::vector<int>>& gpu_res, int & gpu_wcc_pass){
    std::cout << "Checking GPU_WCC results..." << std::endl;
    
    int size = gpu_res.size();

    for (auto &v : gpu_res) {
        if (!v.size()) {
            std::cout << "One of GPU WCC results is empty!" << std::endl;
            return;
        }
        std::sort(v.begin(), v.end());
    }

    std::sort(gpu_res.begin(), gpu_res.end(), compare);

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

    cout<<"gpu_res.size="<<gpu_res.size()<<", base_res.size()="<<base_res.size()<<endl;
    cout<<"gpu_res[0].size="<<gpu_res[0].size()<<", base_res[0].size()="<<base_res[0].size()<<endl;

    for (int i = 0; i < size; i++) {
        if (base_res[i].size() != gpu_res[i].size()) {
            std::cout << "Baseline file and GPU WCC results are not the same!" << std::endl;
            std::cout << "Baseline file: " << graph.vertex_id_to_str[i] << " " << base_res[i][0] << std::endl;
            std::cout << "GPU_WCC result: " << graph.vertex_id_to_str[gpu_res[i][0]] << " " << gpu_res[i][0] << std::endl;
            base_line.close();
            return;
        }
        for (int j = 0; j < base_res[i].size(); j++) {
            if (base_res[i][j] != gpu_res[i][j]) {
                std::cout << "Baseline file and CPU WCC results are not the same!" << std::endl;
                std::cout << "Difference at: " << graph.vertex_id_to_str[base_res[i][j]] << " " << graph.vertex_id_to_str[gpu_res[i][j]] << std::endl;
                base_line.close();
                return;
            }
        }
    }

    std::cout << "GPU_WCC results are correct!" << std::endl;
    gpu_wcc_pass = 1;
    base_line.close();

}

void cdlp_check_gpu(graph_structure<double>& graph, std::vector<int>& gpu_res, int & cdlp_gpu_pass){
    std::cout << "Checking GPU CDLP results..." << std::endl;
    
    int size = gpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of GPU CDLP results is not equal to the number of vertices!" << std::endl;
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
        
        if (gpu_res[v_id] != std::stol(tokens[1])) {
            std::cout << "Baseline file and GPU CDLP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "GPU CDLP result: " << graph.vertex_id_to_str[gpu_res[v_id]] << " " << gpu_res[v_id] << std::endl;
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

    std::cout << "GPU CDLP results are correct!" << std::endl;
    cdlp_gpu_pass = 1;
    base_line.close();
}
