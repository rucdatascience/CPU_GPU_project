
#pragma once

#include "../include/graph_structure/graph_structure.hpp"
#include <vector>
#include <numeric>
#include <iostream>
#include <queue>
#include "../include/ldbc.hpp"

using namespace std;

struct node {
  double dis;//distance from source vertex
  int u;//indicates vertex 

  bool operator>(const node& a) const { return dis > a.dis; }//operator overload
};

std::vector<double> CPU_shortest_paths(std::vector<std::vector<std::pair<int, double>>>& input_graph, int source) {
	//dijkstras-shortest-path-algorithm
	
	double inf = std::numeric_limits<double>::max();

	int N = input_graph.size();

	std::vector<double> distances;
	distances.resize(N, inf); // initial distance from source is inf

	if (source < 0 || source >= N) {
		std::cout << "Invalid source vertex" << std::endl;//Abnormal input judgment
		return distances;
	}

	distances[source] = 0;//Starting distance is 0
	std::vector<int> vis(N, 0);

	std::priority_queue<node, vector<node>, greater<node> > Q;//Using Heap Optimization Algorithm
	Q.push({0, source});

	while (Q.size() > 0) {

		int u = Q.top().u;

		Q.pop();//remove vertex visited this round

		if (vis[u]) continue;//if vertex has already been visited,it shouldn't be pushed to queue again.
		vis[u] = 1;//mark

		for (auto edge : input_graph[u]) {
			//Traverse all adjacent vertexs of a vertex
			int v = edge.first;//vertex pointed by edge
			double w = edge.second;//weight of edge
			//use v to update path cost
			if (distances[v] > distances[u] + w) {
				//If the path cost is smaller, update the new path cost
				distances[v] = distances[u] + w;
				Q.push({distances[v], v});//add new vertex to queue
			}
		}

	}

	return distances;
}

std::map<long long int, double> SSSP_bind_node(LDBC<double> & graph){
	vector<double> ssspVec =  CPU_shortest_paths(graph.OUTs, graph.sssp_src);

    std::map<long long int,   double> strId2value;

    std::vector<long long int> converted_numbers;

    for (const auto& str : graph.vertex_id_to_str) {
        long long int num = std::stoll(str);
        converted_numbers.push_back(num);
    }

    std::sort(converted_numbers.begin(), converted_numbers.end());

	for( int i = 0; i < ssspVec.size(); ++i){
		strId2value.emplace(converted_numbers[i], ssspVec[i]);
    }

	// std::string path = "../data/cpu_bfs_75.txt";
	// storeResult(strId2value, path);//ldbc file

    return strId2value;
}

std::vector<std::string> CPU_shortest_paths_v2(LDBC<double> & graph) {

    std::vector<std::string> ssspVec;
	vector<double> ssspResult =  CPU_shortest_paths(graph.OUTs, graph.sssp_src);

    for(auto & it : ssspResult){
        ssspVec.push_back(std::to_string(it));
    }
    
    return ssspVec;
}