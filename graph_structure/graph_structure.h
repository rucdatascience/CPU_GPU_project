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
#include "parse_string.h"
#include "sorted_vector_binary_operations.h"
#include "binary_save_read_vector_of_vectors.h"

template <typename weight_type>
class CSR_graph;

template <typename weight_type> // weight_type may be int, long long int, float, double...
class graph_structure {
public:
	/*
	this class is for directed and edge-weighted graph
	*/

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
	void load_LDBC(std::string v_path, std::string e_path);
	std::unordered_map<std::string, int> vertex_str_to_id; // vertex_str_to_id[vertex_name] = vertex_id
	std::vector<std::string> vertex_id_to_str; // vertex_id_to_str[vertex_id] = vertex_name

	

	

	

	
	
	


	
};








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
}

template <typename weight_type>
void graph_structure<weight_type>::remove_all_adjacent_edges(int v) {

	for (auto it = OUTs[v].begin(); it != OUTs[v].end(); it++) {
		sorted_vector_binary_operations_erase(OUTs[it->first], v);
	}

	for (auto it = INs[v].begin(); it != INs[v].end(); it++) {
		sorted_vector_binary_operations_erase(INs[it->first], v);
	}

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
	for (auto it : OUTs) {
		num = num + it.size();
	}

	return num;
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
			INs_Edges.push_back(xx.first);
			INs_Edge_weights.push_back(xx.second);
		}
		pointer += INs[i].size();
	}
	INs_Neighbor_start_pointers[V] = pointer;

	pointer = 0;
	for (int i = 0; i < V; i++) {
		ARRAY.OUTs_Neighbor_start_pointers[i] = pointer;
		for (auto& xx : OUTs[i]) {
			OUTs_Edges.push_back(xx.first);
			OUTs_Edge_weights.push_back(xx.second);
		}
		pointer += OUTs[i].size();
	}
	OUTs_Neighbor_start_pointers[V] = pointer;

	return ARRAY;
}











template <typename weight_type>
void graph_structure<weight_type>::add_vertice(std::string line_content) {
	if (vertex_str_to_id.find(line_content) == vertex_str_to_id.end()) {
		vertex_id_to_str.push_back(line_content);
		vertex_str_to_id[line_content] = V++;
	}
}

template <typename weight_type>
void graph_structure<weight_type>::add_edge(std::string e1, std::string e2, weight_type ec) {
	E++;
	add_vertice(e1);
	add_vertice(e2);
	int v1 = vertex_str_to_id[e1];
	int v2 = vertex_str_to_id[e2];
	sorted_vector_binary_operations_insert(ADJs[v1], v2, ec);
	sorted_vector_binary_operations_insert(ADJs_T[v2], v1, ec);
}


template <typename weight_type>
void graph_structure<weight_type>::load_LDBC(std::string v_path, std::string e_path) {
	this->clear();

	std::string line_content;
	std::ifstream myfile(v_path);

	if (myfile.is_open()) {
		while (getline(myfile, line_content))
			add_vertice(line_content);
		myfile.close();
	}
	else {
		std::cout << "Unable to open file " << v_path << std::endl
			<< "Please check the file location or file name." << std::endl;
		getchar();
		exit(1);
	}

	OUTs.resize(V);
	INs.resize(V);

	myfile.open(e_path);

	if (myfile.is_open()) {
		while (getline(myfile, line_content)) {
			std::vector<std::string> Parsed_content = parse_string(line_content, " ");
			int v1 = vertex_id_string2int[Parsed_content[0]];
			int v2 = vertex_id_string2int[Parsed_content[1]];
			weight_type ec = Parsed_content.size() > 2 ? std::stod(Parsed_content[2]) : 1;
			graph_structure<weight_type>::add_edge(v1, v2, ec);
		}
		myfile.close();
	}
	else {
		std::cout << "Unable to open file " << e_path << std::endl
			<< "Please check the file location or file name." << std::endl;
		getchar();
		exit(1);
	}

}
















void graph_structure_example() {

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

	g.txt_save("ss.txt");
	g.txt_read("ss.txt");

	g.print(); // print the graph

	std::cout << "g.size()= " << g.size() << '\n';
}