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

void wcc_result_vs_ldbc(){

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