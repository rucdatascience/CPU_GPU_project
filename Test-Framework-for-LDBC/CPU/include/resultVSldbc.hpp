#pragma once

#include "ldbc.hpp"
#include <algorithm>
#include <cmath>
#include <limits.h>
#include "checkerLDBC.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
/**
void bfs_result_vs_ldbc(LDBC<double> & graph, std::map<long long int, int> & bfs_result, int & is_pass){
    int size = bfs_result.size();

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
        std::istringstream iss(line);
        std::string str_key, str_value;

        if (!(iss >> str_key >> str_value)) {
            std::cerr << "Invalid data row: " << line << std::endl;
            return;
        }

        long long int key;
        long long int value;
        try {
            key = std::stoll(str_key);
            value = std::stoll(str_value);
        } catch (const std::exception& e) {
            std::cerr << "Type conversion error: " << e.what() << std::endl;
            return;
        }

        auto it = bfs_result.find(key);
        if (it != bfs_result.end()) {
            int map_value = it->second;
            if (map_value != value) {
                if(!(it->second == INT_MAX && value == LLONG_MAX)){//Added a correction to the cit data set
                    std::cout << "Line: "<<id+1<<" Baseline node: " << key << " baseline value: " << value << std::endl;
                    std::cout << "Line: "<<id+1<<" Result node: " << it->first << " baseline value: " << it->second << std::endl;
                    return;
                }
            }
        }

        ++id;
    }
   
    std::cout << "BFS results VS baseline is same!" << std::endl;
    is_pass = 1;
    base_line.close();

}

void wcc_result_vs_ldbc(LDBC<double>& graph, std::vector<std::vector<string>>& wcc_result, int & is_pass){
    int size = wcc_result.size();

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
        if (base_res[i].size() != wcc_result[i].size()) {
            std::cout << "Baseline file and GPU WCC results are not the same!" << std::endl;
            std::cout << "Baseline file: " << graph.vertex_id_to_str[i] << " " << base_res[i][0] << std::endl;
            std::cout << "CPU WCC result: " << wcc_result[i][0] << std::endl;
            base_line.close();
            return;
        }
        for (int j = 0; j < base_res[i].size(); j++) {
            if (base_res[i][j] != std::stoi(wcc_result[i][j])) {
                std::cout << "Baseline file and GPU WCC results are not the same!" << std::endl;
                std::cout << "Difference at: " << graph.vertex_id_to_str[base_res[i][j]] << " " << wcc_result[i][j] << std::endl;
                base_line.close();
                return;
            }
        }
    }

    std::cout << "WCC results is same!" << std::endl;
    is_pass = 1;
    base_line.close();


}

void sssp_result_vs_ldbc(LDBC<double> & graph, std::map<long long int, double> & sssp_result, int & is_pass){
    int size = sssp_result.size();

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

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::istringstream iss(line);
        std::string str_key, str_value;

        if (!(iss >> str_key >> str_value)) {
            std::cerr << "Invalid data row: " << line << std::endl;
            return;
        }

        long long int key;
        double value;
        try {
            key = std::stoll(str_key);
            value = std::stod(str_value);
        } catch (const std::exception& e) {
            std::cerr << "Type conversion error: " << e.what() << std::endl;
            return;
        }

        auto it = sssp_result.find(key);
        if (it != sssp_result.end()) {
            double map_value = it->second;
            if (map_value != value) {
                if(fabs(map_value - value)> 1e-4){
                    std::cout << "Line: "<<id+1<<" Baseline node: " << key << " baseline value: " << value << std::endl;
                    std::cout << "Line: "<<id+1<<" Result node: " << it->first << " baseline value: " << it->second << std::endl;
                    return;
                }
            }
        }

        ++id;
    }

    std::cout << "SSSP results VS baseline is same!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void pr_result_vs_ldbc(LDBC<double> & graph, std::map<long long int, double> & pr_result, int & is_pass){
    int size = pr_result.size();

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
        std::istringstream iss(line);
        std::string str_key, str_value;

        if (!(iss >> str_key >> str_value)) {
            std::cerr << "Invalid data row: " << line << std::endl;
            return;
        }

        long long int key;
        double value;
        try {
            key = std::stoll(str_key);
            value = std::stod(str_value);
        } catch (const std::exception& e) {
            std::cerr << "Type conversion error: " << e.what() << std::endl;
            return;
        }

        auto it = pr_result.find(key);
        if (it != pr_result.end()) {
            double map_value = it->second;
            if (map_value != value) {
                if(fabs(map_value - value)> 1e-4){
                    std::cout << "Line: "<<id+1<<" Baseline node: " << key << " baseline value: " << value << std::endl;
                    std::cout << "Line: "<<id+1<<" Result node: " << it->first << " baseline value: " << it->second << std::endl;
                    return;
                }
            }
        }

        ++id;
    }

    std::cout << "PageRank results VS baseline is same!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void cdlp_result_vs_ldbc(LDBC<double> & graph, std::map<long long int, std::string> & cdlp_result, int & is_pass){
    int size = cdlp_result.size();

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
        std::istringstream iss(line);
        std::string str_key, str_value;

        if (!(iss >> str_key >> str_value)) {
            std::cerr << "Invalid data row: " << line << std::endl;
            return;
        }

        long long int key;
        std::string value;
        try {
            key = std::stoll(str_key);
            value = str_value;
        } catch (const std::exception& e) {
            std::cerr << "Type conversion error: " << e.what() << std::endl;
            return;
        }

        auto it = cdlp_result.find(key);
        if (it != cdlp_result.end()) {
            string map_value = it->second;
            if (map_value != value) {
                std::cout << "Line: "<<id+1<<" Baseline node: " << key << " baseline value: " << value << std::endl;
                std::cout << "Line: "<<id+1<<" Result node: " << it->first << " baseline value: " << it->second << std::endl;
                return;
            }
        }

        ++id;
    }

    std::cout << "CDLP results VS baseline is same!" << std::endl;
    is_pass = 1;
    base_line.close();
}
*/

void bfs_ldbc_checker_v2(LDBC<double>& graph, std::vector<std::string>& bfs_res, int & is_pass) {
    
    vector<int> cpu_res;
    
    for (const auto& str : bfs_res) {
        int num = stoi(str);
        cpu_res.push_back(num);
    }

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
            if(!(cpu_res[v_id] == INT_MAX && std::stol(tokens[1]) == LLONG_MAX)){
                std::cout << "Baseline file and GPU BFS results are not the same!" << std::endl;
                std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
                std::cout << "GPU BFS result: " << graph.vertex_id_to_str[v_id] << " " << cpu_res[v_id] << std::endl;
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

    std::cout << "BFS results V2 are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void wcc_ldbc_checker_v2(LDBC<double>& graph, std::vector<std::vector<std::string>>& wcc_res, int & is_pass) {
    
    vector<vector<int>> cpu_res;
    
    for (const auto& innerVec : wcc_res) {
        vector<int> intInnerVec;
        
        for (const string& str : innerVec) {
            // 使用 stoi 或者 stoi 进行字符串到整数的转换
            int num = stoi(str);
            intInnerVec.push_back(num);
        }
        
        cpu_res.push_back(intInnerVec);
    }

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
            std::cout << "CPU WCC result: " << graph.vertex_id_to_str[cpu_res[i][0]] << " " << cpu_res[i][0] << std::endl;
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

    std::cout << "WCC results V2 are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void sssp_ldbc_checker_v2(LDBC<double>& graph, std::vector<std::string>& sssp_res, int & is_pass) {
    
    vector<double> cpu_res;
    
    for (const auto& str : sssp_res) {
        int num = stod(str);
        cpu_res.push_back(num);
    }

    int size = cpu_res.size();

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
            std::cout << "Baseline file and GPU SSSP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "GPU SSSP result: " << graph.vertex_id_to_str[v_id] << " " << cpu_res[v_id] << std::endl;
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
    is_pass = 1;
    base_line.close();
}

void pr_ldbc_checker_v2(LDBC<double>& graph, std::vector<std::string>& pr_res, int & is_pass) {
    
    vector<double> cpu_res;
    
    for (const auto& str : pr_res) {
        int num = stod(str);
        cpu_res.push_back(num);
    }

    int size = cpu_res.size();

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

    std::cout << "PageRank results v2 are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}
