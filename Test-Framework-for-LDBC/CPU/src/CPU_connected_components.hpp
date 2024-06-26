#pragma once

#include "../include/ldbc.hpp"
#include <queue>
#include <vector>

template<typename T> // T is float or double
std::vector<std::vector<int>> CPU_connected_components(std::vector<std::vector<std::pair<int, T>>>& input_graph, std::vector<std::vector<std::pair<int, T>>>& output_graph) {
	//Using BFS method to find connectivity vectors starting from each node
	/*this is to find connected_components using breadth first search; time complexity O(|V|+|E|);
	related content: https://www.boost.org/doc/libs/1_68_0/boost/graph/connected_components.hpp
	https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm*/

	std::vector<std::vector<int>> components;

	/*time complexity: O(V)*/
	int N = input_graph.size();
	std::vector<bool> discovered(N, false);
	//Vector initialization
	for (int i = 0; i < N; i++) {

		if (discovered[i] == false) {
			//If the node has not yet been added to the connected component, search for the connected component starting from the node
			std::vector<int> component;
			/*below is a depth first search; time complexity O(|V|+|E|)*/
			std::queue<int> Q; // Queue is a data structure designed to operate in FIFO (First in First out) context.
			Q.push(i);
			component.push_back(i);
			discovered[i] = true;
			while (Q.size() > 0) {
				int v = Q.front();
				Q.pop(); //Removing that vertex from queue,whose neighbour will be visited now

				for (auto& x : input_graph[v]) {
					int adj_v = x.first;
					if (discovered[adj_v] == false) {
						Q.push(adj_v);
						component.push_back(adj_v);
						discovered[adj_v] = true;
					}
				}
				for (auto& x : output_graph[v]) {
					int adj_v = x.first;
					if (discovered[adj_v] == false) {
						Q.push(adj_v);
						component.push_back(adj_v);
						discovered[adj_v] = true;
					}
				}
			}
			components.push_back(component);
		}
	}
	return components;

}

std::map<std::string, int> getUserWCC(LDBC<double> & graph){
	std::vector<std::vector<int>> wccVec = CPU_connected_components(graph.OUTs, graph.INs);

	std::map<std::string,   int> strId2value;

    return strId2value;
}