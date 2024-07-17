#pragma once

#include <graph_structure/graph_structure.hpp>
#include <queue>
#include <vector>

template<typename T> // T is float or double
std::vector<int> CPU_connected_components(std::vector<std::vector<std::pair<int, T>>>& input_graph, std::vector<std::vector<std::pair<int, T>>>& output_graph) {
	//Using BFS method to find connectivity vectors starting from each node
	/*this is to find connected_components using breadth first search; time complexity O(|V|+|E|);
	related content: https://www.boost.org/doc/libs/1_68_0/boost/graph/connected_components.hpp
	https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm*/

	std::vector<int> parent;

	/*time complexity: O(V)*/
	int N = input_graph.size();
	std::vector<bool> discovered(N, false);
	parent.resize(N);
	//Vector initialization
	for (int i = 0; i < N; i++) {

		if (discovered[i] == false) {
			//If the node has not yet been added to the connected component, search for the connected component starting from the node
			/*below is a depth first search; time complexity O(|V|+|E|)*/
			std::queue<int> Q; // Queue is a data structure designed to operate in FIFO (First in First out) context.
			Q.push(i);
			parent[i] = i;
			discovered[i] = true;
			while (Q.size() > 0) {
				int v = Q.front();
				Q.pop(); //Removing that vertex from queue,whose neighbour will be visited now

				for (auto& x : input_graph[v]) {
					int adj_v = x.first;
					if (discovered[adj_v] == false) {
						Q.push(adj_v);
						parent[adj_v] = parent[v];
						discovered[adj_v] = true;
					}
				}
				for (auto& x : output_graph[v]) {
					int adj_v = x.first;
					if (discovered[adj_v] == false) {
						Q.push(adj_v);
						parent[adj_v] = parent[v];
						discovered[adj_v] = true;
					}
				}
			}
		}
	}
	return parent;
}

std::vector<std::pair<std::string, std::string>> CPU_WCC(graph_structure<double> & graph){
	std::vector<int> wccVec = CPU_connected_components(graph.OUTs, graph.INs);
	return graph.res_trans_id_id(wccVec);
}