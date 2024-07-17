#pragma once

#include <graph_structure/graph_structure.hpp>
#include <algorithm>
#include <cmath>
#include <limits.h>

bool compare(std::vector<int>& a, std::vector<int>& b) {
    return a[0] < b[0];
}

void Bfs_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, int>>& res, bool& is_pass) {

    int size = res.size();

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

    std::vector<int> id_res(graph.V, -1);
    for (auto &p : res)
        id_res[graph.vertex_str_to_id[p.first]] = p.second;

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
        if (id_res[v_id] != std::stol(tokens[1])) {
            if(!(id_res[v_id] == INT_MAX && std::stol(tokens[1]) == LLONG_MAX)){
                std::cout << "Baseline file and GPU BFS results are not the same!" << std::endl;
                std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
                std::cout << "BFS result: " << graph.vertex_id_to_str[v_id] << " " << id_res[v_id] << std::endl;
                base_line.close();
                return;
            }
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "BFS results are correct!" << std::endl;
    is_pass = true;
    base_line.close();
}

void WCC_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, std::string>>& res, bool& is_pass) {
    std::vector<std::vector<int>> temp;
    temp.resize(graph.V);
    for (auto &p : res)
        temp[graph.vertex_str_to_id[p.second]].push_back(graph.vertex_str_to_id[p.first]);
    std::vector<std::vector<int>> components;
    for (int i = 0; i < graph.V; i++) {
        if (temp[i].size() > 0)
            components.push_back(temp[i]);
    }

    int size = components.size();
    for (auto &v : components) {
        if (!v.size()) {
            std::cout << "One of WCC results is empty!" << std::endl;
            return;
        }
        std::sort(v.begin(), v.end());
    }

    std::sort(components.begin(), components.end(), compare);

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
    std::vector<int> base_components;

    base_components.resize(graph.V, 0);

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
        base_components[graph.vertex_str_to_id[tokens[0]]] = graph.vertex_str_to_id[tokens[1]];
    }

    std::vector<std::vector<int>> componentLists(graph.V);

    for (int i = 0; i < graph.V; i++) {
        componentLists[base_components[i]].push_back(i);
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
        if (base_res[i].size() != components[i].size()) {
            std::cout << "Baseline file and WCC results are not the same!" << std::endl;
            std::cout << "Baseline component size is " << base_res[i].size() << std::endl;
            std::cout << "WCC result component size is " << components[i].size() << std::endl;
            return;
        }
        for (int j = 0; j < base_res[i].size(); j++) {
            if (base_res[i][j] != components[i][j]) {
                std::cout << "Baseline file and WCC results are not the same!" << std::endl;
                std::cout << "Difference at: " << graph.vertex_id_to_str[base_res[i][j]] << " " << graph.vertex_id_to_str[components[i][j]] << std::endl;
                base_line.close();
                return;
            }
        }
    }

    std::cout << "WCC results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void SSSP_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, double>>& res, bool& is_pass) {
    
    int size = res.size();

    if (size != graph.V) {
        std::cout << "Size of SSSP results is not equal to the number of vertices!" << std::endl;
        return;
    }

    std::string base_line_file = "../results/" + graph.vertex_file;
    // remove the last two char
    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-SSSP";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    std::vector<double> id_res(graph.V, INT_MAX);

    for (auto &p : res)
        id_res[graph.vertex_str_to_id[p.first]] = p.second;

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

        if (fabs(id_res[v_id] - std::stod(tokens[1])) > 1e-4) {
            std::cout << "Baseline file and SSSP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "SSSP result: " << graph.vertex_id_to_str[v_id] << " " << id_res[v_id] << std::endl;
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

    std::cout << "SSSP results are correct!" << std::endl;
    is_pass = true;
    base_line.close();
}

void PR_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, double>>& res, bool& is_pass) {

    int size = res.size();

    std::vector<double> id_res(graph.V, 0);

    for (auto &p : res)
        id_res[graph.vertex_str_to_id[p.first]] = p.second;

    if (size != graph.V) {
        std::cout << "Size of PageRank results is not equal to the number of vertices!" << std::endl;
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

        if (fabs(id_res[v_id] - std::stod(tokens[1])) > 1e-4) {
            std::cout << "Baseline file and PageRank results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "PageRank result: " << graph.vertex_id_to_str[v_id] << " " << id_res[v_id] << std::endl;
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

    std::cout << "PageRank results are correct!" << std::endl;
    is_pass = true;
    base_line.close();
}

void CDLP_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, std::string>>& res, bool& is_pass) {
    int size = res.size();

    std::vector<std::string> id_res;

    for (auto &p : res)
        id_res.push_back(p.second);

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
        
        if (id_res[v_id] != tokens[1]) {
            std::cout << "Baseline file and CDLP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "CDLP result: " << id_res[v_id] << std::endl;
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

    std::cout << "CDLP results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}
