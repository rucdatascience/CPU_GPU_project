#pragma once

/*the following codes are for testing

---------------------------------------------------
a cpp file (try.cpp) for running the following example code:
----------------------------------------

#include <iostream>
#include <fstream>
using namespace std;

#include <graph_structure/graph_structure.h>


int main()
{
	graph_structure_example();
}

------------------------------------------------------------------------------------------
Commends for running the above cpp file on Linux:

g++ -std=c++17 -I/home/boost_1_75_0 -I/root/CPU_GPU_project try.cpp -lpthread -O3 -o A
./A
rm A

(optional to put the above commends in run.sh, and then use the comment: sh run.sh)


*/

#include <vector>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include "parse_string.hpp"
#include "sorted_vector_binary_operations.hpp"
#include "binary_save_read_vector_of_vectors.hpp"

template <typename weight_type>
class CSR_graph;

template <typename weight_type> // weight_type may be int, long long int, float, double...
class graph_structure {
public:
	/*
	this class is for directed and edge-weighted graph
	*/

	int V = 0; // the number of vertices
	long long E = 0; // the number of edges

	// OUTs[u] = v means there is an edge starting from u to v
	std::vector<std::vector<std::pair<int, weight_type>>> OUTs;
	// INs is transpose of OUTs. INs[u] = v means there is an edge starting from v to u
	std::vector<std::vector<std::pair<int, weight_type>>> INs;

	/*constructors*/
	graph_structure() {}
	graph_structure(int n) {
		V = n;
		OUTs.resize(n); // initialize n vertices
		INs.resize(n);
	}
	int size() {
		return V;
	}

	/*class member functions*/
	inline void add_edge(int, int, weight_type); // this function can change edge weights
	inline void remove_edge(int, int);
	inline void remove_all_adjacent_edges(int);
	inline bool contain_edge(int, int); // whether there is an edge
	inline weight_type edge_weight(int, int);
	inline long long int edge_number(); // the total number of edges
	inline void print();
	inline void clear();
	inline int out_degree(int);
	inline int in_degree(int);
	inline CSR_graph<weight_type> toCSR();

	/* 
	LDBC 

	一开始：给LDBC文件路径、数据名称，以便批处理读取不同LDBC数据

	先读Properties file： 
	是有向图还是无向图 bool；是否权重 bool;
	5个bool变量：List of supported algorithms on the graph；读图后，test每个支持的算子
	BFS、CDLP、PR、SSSP、SSSP的参数；

	先读V，再读E

	LDBC的结果测试方法：      https://www.jianguoyun.com/p/DW-YrpAQvbHvCRiO_bMFIAA      2.4 Output Validation 章节

	*/
	bool is_directed = true;
	bool is_weight = false;
	bool is_sssp_weight = true;

	bool sup_bfs = false;
	bool sup_cdlp = false;
	bool sup_pr = false;
	bool sup_wcc = false;
	bool sup_sssp = false;

	int id = 0;
	int bfs_src = 0;
	int cdlp_max_its = 10;
	int pr_its = 10;
	int sssp_src = 0;
	double pr_damping = 0.85;

	void load_LDBC();
	std::unordered_map<std::string, int> vertex_str_to_id; // vertex_str_to_id[vertex_name] = vertex_id
	std::vector<std::string> vertex_id_to_str; // vertex_id_to_str[vertex_id] = vertex_name

	int add_vertice(std::string);
	void add_edge(std::string, std::string, weight_type);

	std::string vertex_file, edge_file;

	void read_config(std::string config_path);
	
};

/*for GPU*/

template <typename weight_type>
class CSR_graph {
	public:
		std::vector<int> INs_Neighbor_start_pointers, OUTs_Neighbor_start_pointers; // Neighbor_start_pointers[i] is the start point of neighbor information of vertex i in Edges and Edge_weights
		/*
			Now, Neighbor_sizes[i] = Neighbor_start_pointers[i + 1] - Neighbor_start_pointers[i].
			And Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
		*/
		std::vector<int> INs_Edges, OUTs_Edges;  // Edges[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] neighbor IDs
		std::vector<weight_type> INs_Edge_weights, OUTs_Edge_weights; // Edge_weights[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] edge weights
};


/*the following codes are for testing

---------------------------------------------------
a cpp file (try.cpp) for running the following example code:
----------------------------------------

#include <iostream>
#include <fstream>
using namespace std;

#include <graph_structure/graph_structure.h>


int main()
{
	graph_structure_example();
}

------------------------------------------------------------------------------------------
Commends for running the above cpp file on Linux:

g++ -std=c++17 -I/home/boost_1_75_0 -I/root/CPU_GPU_project try.cpp -lpthread -O3 -o A
./A
rm A

(optional to put the above commends in run.sh, and then use the comment: sh run.sh)


*/

/*class member functions*/

template <typename weight_type>
void graph_structure<weight_type>::add_edge(int e1, int e2, weight_type ec) {

	/*we assume that the size of g is larger than e1 or e2;
	 this function can update edge weight; there will be no redundent edge*/

	 /*
	 Add the edges (e1,e2) with the weight ec
	 When the edge exists, it will update its weight.
	 Time complexity:
		 O(log n) When edge already exists in graph
		 O(n) When edge doesn't exist in graph
	 */

	sorted_vector_binary_operations_insert(OUTs[e1], e2, ec);
	sorted_vector_binary_operations_insert(INs[e2], e1, ec);

	if (!is_directed) {
		sorted_vector_binary_operations_insert(OUTs[e2], e1, ec);
		sorted_vector_binary_operations_insert(INs[e1], e2, ec);
	}
}

template <typename weight_type>
void graph_structure<weight_type>::remove_edge(int e1, int e2) {

	/*we assume that the size of g is larger than e1 or e2*/
	/*
	 Remove the edges (e1,e2)
	 If the edge does not exist, it will do nothing.
	 Time complexity: O(n)
	*/

	sorted_vector_binary_operations_erase(OUTs[e1], e2);
	sorted_vector_binary_operations_erase(INs[e2], e1);

	if (!is_directed) {
		sorted_vector_binary_operations_erase(OUTs[e2], e1);
		sorted_vector_binary_operations_erase(INs[e1], e2);
	}
}

template <typename weight_type>
void graph_structure<weight_type>::remove_all_adjacent_edges(int v) {
	for (auto it = OUTs[v].begin(); it != OUTs[v].end(); it++)
		sorted_vector_binary_operations_erase(INs[it->first], v);

	for (auto it = INs[v].begin(); it != INs[v].end(); it++)
		sorted_vector_binary_operations_erase(OUTs[it->first], v);

	std::vector<std::pair<int, weight_type>>().swap(OUTs[v]);
	std::vector<std::pair<int, weight_type>>().swap(INs[v]);
}

template <typename weight_type>
bool graph_structure<weight_type>::contain_edge(int e1, int e2) {

	/*
	Return true if graph contain edge (e1,e2)
	Time complexity: O(logn)
	*/

	return sorted_vector_binary_operations_search(OUTs[e1], e2);
}

template <typename weight_type>
weight_type graph_structure<weight_type>::edge_weight(int e1, int e2) {

	/*
	Return the weight of edge (e1,e2)
	If the edge does not exist, return std::numeric_limits<double>::max()
	Time complexity: O(logn)
	*/

	return sorted_vector_binary_operations_search_weight(OUTs[e1], e2);
}

template <typename weight_type>
long long int graph_structure<weight_type>::edge_number() {

	/*
	Returns the number of edges in the figure
	Time complexity: O(n)
	*/

	long long int num = 0;
	for (auto it : OUTs)
		num = num + it.size();

	return is_directed ? num : num / 2;
}

template <typename weight_type>
void graph_structure<weight_type>::clear() {
	std::vector<std::vector<std::pair<int, weight_type>>>().swap(OUTs);
	std::vector<std::vector<std::pair<int, weight_type>>>().swap(INs);
}

template <typename weight_type>
int graph_structure<weight_type>::out_degree(int v) {
	return OUTs[v].size();
}

template <typename weight_type>
int graph_structure<weight_type>::in_degree(int v) {
	return INs[v].size();
}

template <typename weight_type>
void graph_structure<weight_type>::print() {

	std::cout << "graph_structure_print:" << std::endl;

	for (int i = 0; i < V; i++) {
		std::cout << "Vertex " << i << " OUTs List: ";
		int v_size = OUTs[i].size();
		for (int j = 0; j < v_size; j++) {
			std::cout << "<" << OUTs[i][j].first << "," << OUTs[i][j].second << "> ";
		}
		std::cout << std::endl;
	}
	std::cout << "graph_structure_print END" << std::endl;

}

/*for GPU*/

template <typename weight_type>
CSR_graph<weight_type> graph_structure<weight_type>::toCSR() {

	CSR_graph<weight_type> ARRAY;

	int V = OUTs.size();
	ARRAY.INs_Neighbor_start_pointers.resize(V + 1); // Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
	ARRAY.OUTs_Neighbor_start_pointers.resize(V + 1);

	int pointer = 0;
	for (int i = 0; i < V; i++) {
		ARRAY.INs_Neighbor_start_pointers[i] = pointer;
		for (auto& xx : INs[i]) {
			ARRAY.INs_Edges.push_back(xx.first);
			ARRAY.INs_Edge_weights.push_back(xx.second);
		}
		pointer += INs[i].size();
	}
	ARRAY.INs_Neighbor_start_pointers[V] = pointer;

	pointer = 0;
	for (int i = 0; i < V; i++) {
		ARRAY.OUTs_Neighbor_start_pointers[i] = pointer;
		for (auto& xx : OUTs[i]) {
			ARRAY.OUTs_Edges.push_back(xx.first);
			ARRAY.OUTs_Edge_weights.push_back(xx.second);
		}
		pointer += OUTs[i].size();
	}
	ARRAY.OUTs_Neighbor_start_pointers[V] = pointer;

	return ARRAY;
}

template <typename weight_type>
void graph_structure<weight_type>::read_config(std::string config_path) {
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
			if (parts.back() == "vertex-file") {
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
			else if (parts.back() == "names") {
				if (value == "weight")
					is_weight = true;
				else
					is_weight = false;
				std::cout << "is_weight: " << is_weight << std::endl;
			}
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
					bfs_src = add_vertice(value);
					std::cout << "bfs_source_vertex: " << value << " id: " << bfs_src << std::endl;
				}
				else {
					sssp_src = add_vertice(value);
					std::cout << "sssp_source_vertex: " << value << " id: " << sssp_src << std::endl;
				}
            }
        }
    }
	std::cout << "Done." << std::endl; 
    file.close();
}

template <typename weight_type>
int graph_structure<weight_type>::add_vertice(std::string line_content) {
	if (vertex_str_to_id.find(line_content) == vertex_str_to_id.end()) {
		vertex_id_to_str.push_back(line_content);
		vertex_str_to_id[line_content] = id++;
	}
	return vertex_str_to_id[line_content];
}

template <typename weight_type>
void graph_structure<weight_type>::add_edge(std::string e1, std::string e2, weight_type ec) {
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
void graph_structure<weight_type>::load_LDBC() {
	this->clear();

	std::cout << "Loading vertices..." << std::endl;
	std::string line_content;
	std::ifstream myfile("../data/" + vertex_file);

	if (myfile.is_open()) {
		while (getline(myfile, line_content))
			add_vertice(line_content);
		myfile.close();
	}
	else {
		std::cout << "Unable to open file " << vertex_file << std::endl
			<< "Please check the file location or file name." << std::endl;
		getchar();
		exit(1);
	}

	std::cout << "Done." << std::endl;

	OUTs.resize(V);
	INs.resize(V);

	std::cout << "Loading edges..." << std::endl;
	myfile.open("../data/" + edge_file);

	if (myfile.is_open()) {
		while (getline(myfile, line_content)) {
			std::vector<std::string> Parsed_content = parse_string(line_content, " ");
			int v1 = add_vertice(Parsed_content[0]);
			int v2 = add_vertice(Parsed_content[1]);
			weight_type ec = Parsed_content.size() > 2 ? std::stod(Parsed_content[2]) : 1;
			graph_structure<weight_type>::add_edge(v1, v2, ec);
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

inline void graph_structure_example() {

	/*
	Create a complete graph of 10 nodes
	Weight of edge (u,v) and (v,u) equal to min(u,v)+max(u,v)*0.1
	*/
	using std::cout;
	int N = 10;
	graph_structure<float> g(N);

	/*
	Insert the edge
	When the edge exists, it will update its weight.
	*/
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++) {
			g.add_edge(i, j, j + 0.1 * i); // Insert the edge(i,j) with value j+0.1*i
		}
	}

	/*
	Get the number of edges, (u,v) and (v,u) only be counted once
	The output is 45 (10*9/2)
	*/
	std::cout << g.edge_number() << '\n';

	/*
	Check if graph contain the edge (3,1) and get its value
	The output is 1 1.3
	*/
	std::cout << g.contain_edge(3, 1) << " " << g.edge_weight(3, 1) << '\n';

	/*
	Remove half of the edge
	If the edge does not exist, it will do nothing.
	*/
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++) {
			if ((i + j) % 2 == 1)
				g.remove_edge(i, j);
		}
	}

	/*
	Now the number of edges is 20
	*/
	std::cout << g.edge_number() << '\n';;

	/*
	Now the graph no longer contain the edge (3,0) and its value become std::numeric_limits<double>::max()
	*/
	std::cout << g.contain_edge(3, 0) << " " << g.edge_weight(3, 0) << '\n';

	g.print(); // print the graph

	g.remove_all_adjacent_edges(1);

	g.print(); // print the graph

	std::cout << "g.size()= " << g.size() << '\n';
}
