#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <stdexcept>
#include "../include/graph_structure/parse_string.hpp"
#include "../include/graph_structure/sorted_vector_binary_operations.hpp"
#include "../include/graph_structure/binary_save_read_vector_of_vectors.hpp"

/* 
	LDBC test process
	At the beginning: give LDBC file path, data name, so that batch read different LDBC data
	step1: Read the Properties file first: determine whether the graph is directed or undirected bool; Whether weight bool;
	the graph contains 5 bool variables: List of supported algorithms on the graph;
	After reading the graph, test the parameters of each supported operator BFS, CDLP, PR, SSSP, and SSSP;
	Read V first, then E
	LDBC the results of the test method: https://www.jianguoyun.com/p/DW-YrpAQvbHvCRiO_bMFIAA
	For the verification method, see 2.4 Output Validation
*/

template <typename weight_type> // weight_type may be int, long long int, float, double...
class LDBC {
public:
	
	int V = 0; // the number of vertices
	long long E = 0; // the number of edges

	// OUTs[u] = v means there is an edge starting from u to v
	std::vector<std::vector<std::pair<int, weight_type>>> OUTs;
	// INs is transpose of OUTs. INs[u] = v means there is an edge starting from v to u
	std::vector<std::vector<std::pair<int, weight_type>>> INs;

	LDBC() {}
	LDBC(int n) {
		V = n;
		OUTs.resize(n); // initialize n vertices
		INs.resize(n);
	}
	int size() {
		return V;
	}

	int add_vertice(std::string);//Read the vertex information in the ldbc file as a string
	inline void add_edge(int, int, weight_type); // this function can change edge weights	
	void add_edge(std::string, std::string, weight_type);
	inline void print();//print graph
	inline void clear();// clear graph
	
	void load_LDBC();
	void read_config(std::string config_path);


	bool is_directed = true;//direct graph or undirect graph
	bool is_weight = false;// weight graph or no weight graph
	bool is_sssp_weight = true;//the weight of sssp

	bool sup_bfs = false;
	bool sup_cdlp = false;
	bool sup_pr = false;
	bool sup_wcc = false;
	bool sup_sssp = false;

	std::string bfs_src_name;//get bfs vertex source
	std::string sssp_src_name;//get sssp vertex source

	int id = 0;
	int bfs_src = 0;//define bfs vertex source is 0
	int cdlp_max_its = 10;//cdlp algo max iterator num
	int pr_its = 10;//pr algo iterator num
	int sssp_src = 0;//define sssp vertex source is 0
	double pr_damping = 0.85;//pr algorithm damping coefficient

	std::unordered_map<std::string, int> vertex_str_to_id; // vertex_str_to_id[vertex_name] = vertex_id
	std::vector<std::string> vertex_id_to_str; // vertex_id_to_str[vertex_id] = vertex_name
	std::string vertex_file, edge_file;

	
	
};


template <typename weight_type>
int LDBC<weight_type>::add_vertice(std::string line_content) {
	if (vertex_str_to_id.find(line_content) == vertex_str_to_id.end()) {
		vertex_id_to_str.push_back(line_content);
		vertex_str_to_id[line_content] = id++;//Read the LDBC file and renumber it from 0
	}
	return vertex_str_to_id[line_content];//the ldbc vertex file lineNo is the vertex matrix size
}

template <typename weight_type>
void LDBC<weight_type>::add_edge(int e1, int e2, weight_type ec) {

	sorted_vector_binary_operations_insert(OUTs[e1], e2, ec);
	sorted_vector_binary_operations_insert(INs[e2], e1, ec);

	if (!is_directed) {
		sorted_vector_binary_operations_insert(OUTs[e2], e1, ec);
		sorted_vector_binary_operations_insert(INs[e1], e2, ec);
	}
}

template <typename weight_type>
void LDBC<weight_type>::add_edge(std::string e1, std::string e2, weight_type ec) {
	E++;
	int v1 = add_vertice(e1);
	int v2 = add_vertice(e2);
	sorted_vector_binary_operations_insert(INs[v1], v2, ec);
	sorted_vector_binary_operations_insert(OUTs[v2], v1, ec);
	if (!is_directed) {
		sorted_vector_binary_operations_insert(INs[v2], v1, ec);
		sorted_vector_binary_operations_insert(OUTs[v1], v2, ec);
	}
}

template <typename weight_type>
void LDBC<weight_type>::read_config(std::string config_path) {
	std::ifstream file(config_path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << config_path << std::endl;
        return;
    }

	std::cout << "Reading config file..." << std::endl;

    while (getline(file, line)) {
		if (line.empty() || line[0] == '#')
			continue;

		auto split_str = parse_string(line, " = ");

		if (split_str.size() != 2) {
			std::cerr << "Invalid line: " << line << std::endl;
			continue;
		}

        auto key = split_str[0];
		auto value = split_str[1];

        auto parts = parse_string(key, ".");
        if (parts.size() >= 2) {
			if (parts.back() == "vertex-file") {//Reading *.properties file to get vertex file. eg. datagen-7_5-fb.v
				vertex_file = value;
				std::cout << "vertex_file: " << vertex_file << std::endl;
			}
			else if (parts.back() == "edge-file") {
				edge_file = value;
				std::cout << "edge_file: " << edge_file << std::endl;
			}
			else if (parts.back() == "vertices") {
				V = stoi(value);
				std::cout << "V: " << V << std::endl;
			}
			else if (parts.back() == "edges") {
				E = stoi(value);
				std::cout << "E: " << E << std::endl;
			}
			else if (parts.back() == "directed") {
				if (value == "false")
					is_directed = false;
				else
					is_directed = true;
				std::cout << "is_directed: " << is_directed << std::endl;
			}
			else if (parts.back() == "names") {//eg. graph.datagen-7_5-fb.edge-properties.names = weight
				if (value == "weight")
					is_weight = true;
				else
					is_weight = false;
				std::cout << "is_weight: " << is_weight << std::endl;
			}//Gets the type of algorithm contained in the configuration file
			else if (parts.back() == "algorithms") {
				auto algorithms = parse_string(value, ", ");
				for (auto& algorithm : algorithms) {
					if (algorithm == "bfs")
						sup_bfs = true;
					else if (algorithm == "cdlp")
						sup_cdlp = true;
					else if (algorithm == "pr")
						sup_pr = true;
					else if (algorithm == "sssp")
						sup_sssp = true;
					else if (algorithm == "wcc")
						sup_wcc = true;
				}
				std::cout << "bfs: " << sup_bfs << std::endl;
				std::cout << "cdlp: " << sup_cdlp << std::endl;
				std::cout << "pr: " << sup_pr << std::endl;
				std::cout << "sssp: " << sup_sssp << std::endl;
				std::cout << "wcc: " << sup_wcc << std::endl;
			}
			else if (parts.back() == "cdlp-max-iterations") {
				cdlp_max_its = stoi(value);
				std::cout << "cdlp_max_its: " << cdlp_max_its << std::endl;
			}
			else if (parts.back() == "pr-damping-factor") {
				pr_damping = stod(value);
				std::cout << "pr_damping: " << pr_damping << std::endl;
			}
			else if (parts.back() == "pr-num-iterations") {
				pr_its = stoi(value);
				std::cout << "pr_its: " << pr_its << std::endl;
			}
			else if (parts.back() == "sssp-weight-property") {
				if (value == "weight")
					is_sssp_weight = true;
				else
					is_sssp_weight = false;
				std::cout << "is_sssp_weight: " << is_sssp_weight << std::endl;
			}
			else if (parts.back() == "max-iterations") {
				cdlp_max_its = stoi(value);
				std::cout << "cdlp_max_its: " << cdlp_max_its << std::endl;
			}
			else if (parts.back() == "damping-factor") {
				pr_damping = stod(value);
				std::cout << "pr_damping: " << pr_damping << std::endl;
			}
			else if (parts.back() == "num-iterations") {
				pr_its = stoi(value);
				std::cout << "pr_its: " << pr_its << std::endl;
			}
			else if (parts.back() == "weight-property") {
				if (value == "weight")
					is_sssp_weight = true;
				else
					is_sssp_weight = false;
				std::cout << "is_sssp_weight: " << is_sssp_weight << std::endl;
			}
            else if (parts.back() == "source-vertex") {
				if (parts[parts.size() - 2] == "bfs") {
					bfs_src_name = value;//get bfs source vertex; eg. graph.datagen-7_5-fb.bfs.source-vertex = 6
					std::cout << "bfs_source_vertex: " << value << std::endl;
				}
				else {
					sssp_src_name = value;//get sssp source vertex; eg. graph.datagen-7_5-fb.sssp.source-vertex = 6
					std::cout << "sssp_source_vertex: " << value  << std::endl;
				}
            }
        }
    }
	std::cout << "Done." << std::endl; 
    file.close();
}

template <typename weight_type>
void LDBC<weight_type>::load_LDBC() {
	this->clear();

	std::cout << "Loading vertices..." << std::endl;
	std::string line_content;
	std::ifstream myfile("../data/" + vertex_file);

	if (myfile.is_open()) {
		while (getline(myfile, line_content))//read data line by line
			add_vertice(line_content);//Parsed the read data
		myfile.close();
	}
	else {
		std::cout << "Unable to open file " << vertex_file << std::endl
			<< "Please check the file location or file name." << std::endl;
		getchar();
		exit(1);
	}

	std::cout << "Done." << std::endl;
	if (sup_bfs) {
		if (vertex_str_to_id.find(bfs_src_name) == vertex_str_to_id.end()) {//bfs_src_name from read_configure
			std::cout << "Invalid source vertex for BFS" << std::endl;
			getchar();
			exit(1);
		}
		else
			bfs_src = vertex_str_to_id[bfs_src_name];
	}
		
	if (sup_sssp) {
		if (vertex_str_to_id.find(sssp_src_name) == vertex_str_to_id.end()) {//sssp_src_name from read_configure
			std::cout << "Invalid source vertex for SSSP" << std::endl;
			getchar();
			exit(1);
		}
		else
			sssp_src = vertex_str_to_id[sssp_src_name];
	}

	OUTs.resize(V);
	INs.resize(V);

	std::cout << "Loading edges..." << std::endl;
	myfile.open("../data/" + edge_file);

	if (myfile.is_open()) {
		while (getline(myfile, line_content)) {
			std::vector<std::string> Parsed_content = parse_string(line_content, " ");
			int v1 = add_vertice(Parsed_content[0]);//get 1st vertex
			int v2 = add_vertice(Parsed_content[1]);//get 2nd vertex
			weight_type ec = Parsed_content.size() > 2 ? std::stod(Parsed_content[2]) : 1;//get weight
			LDBC<weight_type>::add_edge(v1, v2, ec);
		}
		myfile.close();
	}
	else {
		std::cout << "Unable to open file " << edge_file << std::endl
			<< "Please check the file location or file name." << std::endl;
		getchar();
		exit(1);
	}
	std::cout << "Done." << std::endl;
}

template <typename weight_type>
void LDBC<weight_type>::clear() {
	std::vector<std::vector<std::pair<int, weight_type>>>().swap(OUTs);
	std::vector<std::vector<std::pair<int, weight_type>>>().swap(INs);
}

template <typename weight_type>
void LDBC<weight_type>::print() {

	std::cout << "LDBC_print:" << std::endl;

	for (int i = 0; i < V; i++) {
		std::cout << "Vertex " << i << " OUTs List: ";
		int v_size = OUTs[i].size();
		for (int j = 0; j < v_size; j++) {
			std::cout << "<" << OUTs[i][j].first << "," << OUTs[i][j].second << "> ";
		}
		std::cout << std::endl;
	}
	std::cout << "LDBC_print END" << std::endl;

}

// void storeResult(std::map<long long int, int> & strId2value, std::string & path){
//     std::ofstream outFile(path);

//     if (outFile.is_open()) {
//         for (const auto& pair : strId2value) {
//             outFile << pair.first << " " << pair.second << std::endl;
//         }
//         outFile.close();
//         std::cout << "File write complete!" << std::endl;
//     } else {
//         std::cerr << "Unable to open file!" << std::endl;
//     }

// }
