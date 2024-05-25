#pragma once

#include <graph_structure/graph_structure.hpp>
#include <list>
#include <queue>
#include <vector>

template<typename T> // T is float or double
std::list<std::list<int>> CPU_connected_components(std::vector<std::vector<std::pair<int, T>>>& input_graph) {
	//Using BFS method to find connectivity vectors starting from each node
	/*this is to find connected_components using breadth first search; time complexity O(|V|+|E|);
	related content: https://www.boost.org/doc/libs/1_68_0/boost/graph/connected_components.hpp
	https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm*/

	std::list<std::list<int>> components;
	//Create linked list elements int for linked lists
	/*time complexity: O(V)*/
	int N = input_graph.size();
	
	std::vector<bool> discovered(N, false);
	//Vector initialization
	for (int i = 0; i < N; i++) {

		if (discovered[i] == false) {
			//If the node has not yet been added to the connected component, search for the connected component starting from the node
			std::list<int> component;
			/*below is a depth first search; time complexity O(|V|+|E|)*/
			std::queue<int> Q; // Queue is a data structure designed to operate in FIFO (First in First out) context.
			Q.push(i);
			component.push_back(i);
			discovered[i] = true;
			while (Q.size() > 0) { 
				int v = Q.front();
				Q.pop(); //Removing that vertex from queue,whose neighbour will be visited now

				int adj_size = input_graph[v].size();
				for (int j = 0; j < adj_size; j++) {
					//Traverse all adjacent points of node v, if not accessed, join the queue and set the access flag
					int adj_v = input_graph[v][j].first;
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








