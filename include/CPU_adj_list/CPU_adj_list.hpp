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
#include <algorithm>

#include <CPU_adj_list/parse_string.hpp>
#include <CPU_adj_list/sorted_vector_binary_operations.hpp>
#include <CPU_adj_list/binary_save_read_vector_of_vectors.hpp>

template <typename weight_type> // weight_type may be int, long long int, float, double...
class graph_structure {
public:
	/*
	this class is for directed and edge-weighted graph
	*/

	int V = 0; // the number of vertices
	long long E = 0; // the number of edges

	bool is_directed = true;//direct graph or undirect graph
	bool is_weight = false;// weight graph or no weight graph
	bool is_sssp_weight = true;//the weight of sssp

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
	inline void fast_add_edge(int, int, weight_type);
	inline void remove_edge(int, int);//Remove any edge that connects two vertices
	inline void remove_all_adjacent_edges(int);//Remove all edges, the input params is vertex numbers
	inline bool contain_edge(int, int); // whether there is an edge
	inline weight_type edge_weight(int, int); //get edge weight
	inline long long int edge_number(); // the total number of edges
	inline void print();//print graph
	inline void clear();// clear graph
	inline int out_degree(int);//get graph out degree
	inline int in_degree(int);//get graph in degree

	int id = 0;
    std::unordered_map<std::string, int> vertex_str_to_id; // vertex_str_to_id[vertex_name] = vertex_id
	std::vector<std::string> vertex_id_to_str; // vertex_id_to_str[vertex_id] = vertex_name

	int add_vertice(std::string);//Read the vertex information in the ldbc file as a string
	void add_edge(std::string, std::string, weight_type);

	template <typename T>
	std::vector<std::pair<std::string, T>> res_trans_id_val(std::vector<T>& res);
	std::vector<std::pair<std::string, std::string>> res_trans_id_id(std::vector<int>& wcc_res);
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

template <typename weight_type>
void graph_structure<weight_type>::fast_add_edge(int e1, int e2, weight_type ec) {
	E++;
	INs[e1].push_back(std::make_pair(e2, ec));
	OUTs[e2].push_back(std::make_pair(e1, ec));
	if (!is_directed) {
		INs[e2].push_back(std::make_pair(e1, ec));
		OUTs[e1].push_back(std::make_pair(e2, ec));
	}
}

//graph_structure test
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

//Store the vertex data in the ldbc into a vector
template <typename weight_type>
int graph_structure<weight_type>::add_vertice(std::string line_content) {
	if (vertex_str_to_id.find(line_content) == vertex_str_to_id.end()) {
		vertex_id_to_str.push_back(line_content);
		vertex_str_to_id[line_content] = id++;
	}
	return vertex_str_to_id[line_content];
}

//Code rewrite. Attention please the data type of input paramers.
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
template <typename T>
std::vector<std::pair<std::string, T>> graph_structure<weight_type>::res_trans_id_val(std::vector<T>& res) {
	std::vector<std::pair<std::string, T>> res_str;
	int res_size = res.size();
	for (int i = 0; i < res_size; i++) {
		res_str.push_back(std::make_pair(vertex_id_to_str[i], res[i]));
	}

	return res_str;
}

template <typename weight_type>
std::vector<std::pair<std::string, std::string>> graph_structure<weight_type>::res_trans_id_id(std::vector<int>& wcc_res) {
	std::vector<std::pair<std::string, std::string>> res_str;
	int res_size = wcc_res.size();
	for (int i = 0; i < res_size; i++)
		res_str.push_back(std::make_pair(vertex_id_to_str[i], vertex_id_to_str[wcc_res[i]]));

	return res_str;
}
