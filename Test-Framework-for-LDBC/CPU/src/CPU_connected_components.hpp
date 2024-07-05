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

std::vector<std::vector<std::string>> CPU_connected_components_v2(LDBC<double> & graph){
	std::vector<std::vector<int>> wccVec = CPU_connected_components(graph.OUTs, graph.INs);

    std::vector<std::vector<std::string>> cpu_wcc_result_v2;

    for (const auto& inner_vec : wccVec) {
            std::vector<std::string> inner_result;
            for (int value : inner_vec) {
                inner_result.push_back(std::to_string(value)); 
            }
            cpu_wcc_result_v2.push_back(inner_result);
    }

    return cpu_wcc_result_v2;

}