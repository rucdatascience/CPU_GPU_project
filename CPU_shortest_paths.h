
#pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <unordered_map>
#include <boost/heap/fibonacci_heap.hpp> 
#include <graph_v_of_v/graph_v_of_v.h>

using namespace std;


struct graph_v_of_v_node_for_sp {
	int index;
	double priority_value;
}; // define the node in the queue
bool operator<(graph_v_of_v_node_for_sp const& x, graph_v_of_v_node_for_sp const& y) {
	return x.priority_value > y.priority_value; // < is the max-heap; > is the min heap
}
typedef typename boost::heap::fibonacci_heap<graph_v_of_v_node_for_sp>::handle_type handle_t_for_graph_v_of_v_sp;


template<typename T> // T is float or double
void CPU_shortest_paths(graph_v_of_v<T>& input_graph, int source, std::vector<T>& distances, std::vector<int>& predecessors) {

	/*Dijkstra's shortest path algorithm: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
	time complexity: O(|E|+|V|log|V|);
	the output distances and predecessors only contain vertices connected to source*/

	T inf = std::numeric_limits<T>::max();

	int N = input_graph.ADJs.size();
	distances.resize(N, inf); // initial distance from source is inf
	predecessors.resize(N);
	std::iota(std::begin(predecessors), std::end(predecessors), 0); // initial predecessor of each vertex is itself

	graph_v_of_v_node_for_sp node;
	boost::heap::fibonacci_heap<graph_v_of_v_node_for_sp> Q;
	std::vector<T> Q_keys(N, inf); // if the key of a vertex is inf, then it is not in Q yet
	std::vector<handle_t_for_graph_v_of_v_sp> Q_handles(N);

	/*initialize the source*/
	Q_keys[source] = 0;
	node.index = source;
	node.priority_value = 0;
	Q_handles[source] = Q.push(node);

	/*time complexity: O(|E|+|V|log|V|) based on fibonacci_heap, not on pairing_heap, which is O((|E|+|V|)log|V|)*/
	while (Q.size() > 0) {

		int top_v = Q.top().index;
		T top_key = Q.top().priority_value;

		Q.pop();

		distances[top_v] = top_key; // top_v is touched

		for (auto it = input_graph.ADJs[top_v].begin(); it != input_graph.ADJs[top_v].end(); it++) {
			int adj_v = it->first;
			T ec = it->second;
			if (Q_keys[adj_v] == inf) { // adj_v is not in Q yet
				Q_keys[adj_v] = top_key + ec;
				node.index = adj_v;
				node.priority_value = Q_keys[adj_v];
				Q_handles[adj_v] = Q.push(node);
				predecessors[adj_v] = top_v;
			}
			else { // adj_v is in Q
				if (Q_keys[adj_v] > top_key + ec) { // needs to update key
					Q_keys[adj_v] = top_key + ec;
					node.index = adj_v;
					node.priority_value = Q_keys[adj_v];
					Q.update(Q_handles[adj_v], node);
					predecessors[adj_v] = top_v;
				}
			}
		}

	}

}