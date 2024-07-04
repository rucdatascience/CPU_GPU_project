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

std::vector<std::vector<std::string>> getUserWCC(LDBC<double> & graph){
	std::vector<std::vector<int>> wccVec = CPU_connected_components(graph.OUTs, graph.INs);

	std::vector<std::vector<std::string>> componentLists;
	
	for(int i = 0; i < wccVec.size(); ++i){
    	std::vector<std::string> component;
		for(int j = 0; j < wccVec[i].size(); ++j){
			std::string vertex_name = graph.vertex_id_to_str[wccVec[i][j]];
			component.push_back(vertex_name);
		}
		componentLists.push_back(component);
	}


	//sort result
    // for (auto& vec : componentLists) {
    //     std::sort(vec.begin(), vec.end(), [](const std::string& a, const std::string& b) {
    //         return std::stoll(a) < std::stoll(b);
    //     });
    // }


    return componentLists;
}