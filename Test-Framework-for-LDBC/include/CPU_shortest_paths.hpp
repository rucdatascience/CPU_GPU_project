
#pragma once

#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <numeric>
#include <iostream>
#include <queue>

using namespace std;

struct node {
  double dis;
  int u;

  bool operator>(const node& a) const { return dis > a.dis; }
};

std::vector<double> CPU_shortest_paths(std::vector<std::vector<std::pair<int, double>>>& input_graph, int source) {

	double inf = std::numeric_limits<double>::max();

	int N = input_graph.size();

	std::vector<double> distances;
	distances.resize(N, inf); // initial distance from source is inf

	if (source < 0 || source >= N) {
		std::cout << "Invalid source vertex" << std::endl;
		return distances;
	}

	distances[source] = 0;
	std::vector<int> vis(N, 0);

	std::priority_queue<node, vector<node>, greater<node> > Q;
	Q.push({0, source});

	while (Q.size() > 0) {

		int u = Q.top().u;

		Q.pop();

		if (vis[u]) continue;
		vis[u] = 1;

		for (auto edge : input_graph[u]) {
			int v = edge.first;
			double w = edge.second;
			if (distances[v] > distances[u] + w) {
				distances[v] = distances[u] + w;
				Q.push({distances[v], v});
			}
		}

	}

	return distances;
}