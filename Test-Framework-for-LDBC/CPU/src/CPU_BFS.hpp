#pragma once

#include <queue>
#include <climits>
#include "../include/ldbc.hpp"
#include <map>
#include <algorithm>
#include <stdexcept>

template<typename T> // T is float or double
std::vector<int> CPU_BFS(std::vector<std::vector<std::pair<int, T>>>& input_graph, int root = 0, int min_depth = 0, int max_depth = INT_MAX) {

	int N = input_graph.size();

	std::vector<int> depth(N, INT_MAX);
	depth[root] = 0;
	//Prevent duplicate traversal while recording depth
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
			//Traversing node v in the graph yields a pair value, adjfirst being the adjacency point
			for (auto& adj : input_graph[v]) { /*processing all the neighbours of v*/
				if (depth[adj.first] > depth[v] + 1) {
					//If the depth of adjacent points is greater, add them to the queue. Otherwise, it means that the adjacent points have already been traversed before
					depth[adj.first] = depth[v] + 1;
					Q.push(adj.first);
				}
			}
		}
	}

	return depth;
}


std::map<long long int,  int> BFS_bind_node(LDBC<double> & graph){
	std::vector<int> bfsValue = CPU_BFS(graph.OUTs, graph.bfs_src, 0, INT_MAX);
	
	std::map<long long int,   int> strId2value;

    std::vector<long long int> converted_numbers;

    for (const auto& str : graph.vertex_id_to_str) {
        long long int num = std::stoll(str);
        converted_numbers.push_back(num);
    }

    std::sort(converted_numbers.begin(), converted_numbers.end());

	for( int i = 0; i < bfsValue.size(); ++i){
		strId2value.emplace(converted_numbers[i], bfsValue[i]);
    }

	// std::string path = "../data/cpu_bfs_75.txt";
	// storeResult(strId2value, path);//ldbc file

    return strId2value;
}