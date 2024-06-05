#pragma once

#include <graph_structure/graph_structure.hpp>
#include <algorithm>
#include <cmath>
#include <checker.hpp>

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
        
        if (cpu_res[v_id] != std::stoi(tokens[1])) {
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