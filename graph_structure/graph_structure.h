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
class ARRAY_graph;

template <typename weight_type> // weight_type may be int, long long int, float, double...
class graph_structure {
public:
	/*
	this class is for directed and edge-weighted graph
	*/

	int V = 0; // number of vertices
	long long int E = 0; // number of edges

	// OUTs[u] = v means there is an edge starting from u to v
	std::vector<std::vector<std::pair<int, weight_type>>> OUTs;
	// INs is transpose of OUTs. INs[u] = v means there is an edge starting from v to u
	std::vector<std::vector<std::pair<int, weight_type>>> INs;

	std::unordered_map<std::string, int> vertex_id_string2int; // vertex_id_string2int[vertex_name] = vertex_id
	std::unordered_map<int, std::string> vertex_id_int2string;

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

	void load_LDBC(std::string v_path, std::string e_path);

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
	inline int search_adjv_by_weight(int, weight_type); // return first e2 such that w(e1,e2) = ec; if there is no such e2, return -1
	inline void txt_save(std::string);
	inline void txt_read(std::string);
	inline void binary_save(std::string);
	inline void binary_read(std::string);
	inline ARRAY_graph<weight_type> toARRAY();
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

	E++;
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

	E--;
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

	E -= OUTs[v].size();
	E -= INs[v].size();

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
	return E;
}

template <typename weight_type>
void graph_structure<weight_type>::print() {

	std::cout << "graph_structure_print:" << std::endl;

	for (int i = 0; i < V; i++) {
		std::cout << "Vertex " << i << " Adj List: ";
		int v_size = OUTs[i].size();
		for (int j = 0; j < v_size; j++) {
			std::cout << "<" << OUTs[i][j].first << "," << OUTs[i][j].second << "> ";
		}
		std::cout << std::endl;
	}
	std::cout << "graph_structure_print END" << std::endl;

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
void graph_structure<weight_type>::txt_save(std::string save_name) {

	std::ofstream outputFile;
	outputFile.precision(10);
	outputFile.setf(std::ios::fixed);
	outputFile.setf(std::ios::showpoint);
	outputFile.open(save_name);

	outputFile << "|V|= " << V << std::endl;
	outputFile << "|E|= " << E << std::endl;
	outputFile << std::endl;

	for (int i = 0; i < V; i++) {
		int v_size = OUTs[i].size();
		for (int j = 0; j < v_size; j++) {
			if (i < OUTs[i][j].first) {
				outputFile << "Edge " << i << " " << OUTs[i][j].first << " " << OUTs[i][j].second << '\n';
			}
		}
	}
	outputFile << std::endl;

	outputFile << "EOF" << std::endl;
}

template <typename weight_type>
void graph_structure<weight_type>::txt_read(std::string save_name) {

	graph_structure<weight_type>::clear();

	std::string line_content;
	std::ifstream myfile(save_name); // open the file
	if (myfile.is_open()) // if the file is opened successfully
	{
		while (getline(myfile, line_content)) // read file line by line
		{
			std::vector<std::string> Parsed_content = parse_string(line_content, " ");

			if (!Parsed_content[0].compare("|V|=")) // when it's equal, compare returns 0
			{
				V = std::stoi(Parsed_content[1]);
				OUTs.resize(V);
				INs.resize(V);
			}
			else if (!Parsed_content[0].compare("Edge"))
			{
				int v1 = std::stoi(Parsed_content[1]);
				int v2 = std::stoi(Parsed_content[2]);
				weight_type ec = std::stod(Parsed_content[3]);
				graph_structure<weight_type>::add_edge(v1, v2, ec);
			}
		}

		myfile.close(); //close the file
	}
	else
	{
		std::cout << "Unable to open file " << save_name << std::endl
			<< "Please check the file location or file name." << std::endl; // throw an error message
		getchar(); // keep the console window
		exit(1); // end the program
	}
}

template <typename weight_type>
void graph_structure<weight_type>::load_LDBC(std::string v_path, std::string e_path) {
	this->clear();

	std::string line_content;
	std::ifstream myfile(v_path);

	if (myfile.is_open()) {
		while (getline(myfile, line_content)) {
			if (vertex_id_string2int.find(line_content) == vertex_id_string2int.end())
				vertex_id_string2int[line_content] = V++;
		}
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

template <typename weight_type>
void graph_structure<weight_type>::binary_save(std::string save_name) {

	binary_save_vector_of_vectors(save_name, OUTs);
}

template <typename weight_type>
void graph_structure<weight_type>::binary_read(std::string save_name) {

	binary_read_vector_of_vectors(save_name, OUTs);
}

template <typename weight_type>
class ARRAY_graph {
public:
	std::vector<int> Neighbor_start_pointers; // Neighbor_start_pointers[i] is the start point of neighbor information of vertex i in Edges and Edge_weights
	std::vector<int> Neighbor_sizes; // Neighbor_sizes[i] is the number of neighbors of vertex i
	/*
		Now, Neighbor_sizes[i] = Neighbor_start_pointers[i + 1] - Neighbor_start_pointers[i].
		And Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
	*/
	std::vector<int> Edges;  // Edges[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] neighbor IDs
	std::vector<weight_type> Edge_weights; // Edge_weights[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] edge weights
};

template <typename weight_type>
ARRAY_graph<weight_type> graph_structure<weight_type>::toARRAY() {

	ARRAY_graph<weight_type> ARRAY;
	auto& Neighbor_start_pointers = ARRAY.Neighbor_start_pointers;
	auto& Neighbor_sizes = ARRAY.Neighbor_sizes;
	auto& Edges = ARRAY.Edges;
	auto& Edge_weights = ARRAY.Edge_weights;

	Neighbor_start_pointers.resize(V + 1);
	Neighbor_sizes.resize(V);

	int pointer = 0;
	for (int i = 0; i < V; i++) {
		Neighbor_start_pointers[i] = pointer;
		Neighbor_sizes[i] = OUTs[i].size();
		for (auto& xx : OUTs[i]) {
			Edges.push_back(xx.first);
			Edge_weights.push_back(xx.second);
		}
		pointer += OUTs[i].size();
	}
	Neighbor_start_pointers[V] = pointer;

	return ARRAY;
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