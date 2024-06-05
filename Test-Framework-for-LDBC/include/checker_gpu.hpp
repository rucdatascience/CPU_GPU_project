#pragma once

#include <graph_structure/graph_structure.hpp>
#include <algorithm>
#include <cmath>
#include <checker.hpp>

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
        
        if (gpu_res[v_id] != std::stoi(tokens[1])) {
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
