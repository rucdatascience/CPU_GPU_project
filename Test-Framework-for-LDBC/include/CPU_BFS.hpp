#pragma once

#include <graph_structure/graph_structure.hpp>
#include <queue>

template<typename T> // T is float or double
std::vector<int> CPU_BFS(std::vector<std::vector<std::pair<int, T>>>& input_graph, int root = 0, int min_depth = 0, int max_depth = INT_MAX) {

	int N = input_graph.size();

	std::vector<int> depth(N, INT_MAX);
	depth[root] = 0;

	std::vector<int> searched_vertices;

	std::queue<int> Q; // Queue is a data structure designed to operate in FIFO (First in First out) context.
	Q.push(root);
	while (Q.size() > 0) {
		int v = Q.front();
		if (depth[v] >= min_depth && depth[v] <= max_depth) {
			searched_vertices.push_back(v);
		}
		Q.pop(); //Removing that vertex from queue,whose neighbour will be visited now

		if (depth[v] + 1 <= max_depth) {
			for (auto& adj : input_graph[v]) { /*processing all the neighbours of v*/
				if (depth[adj.first] > depth[v] + 1) {
					depth[adj.first] = depth[v] + 1;
					Q.push(adj.first);
				}
			}
		}
	}

	return searched_vertices;
}



