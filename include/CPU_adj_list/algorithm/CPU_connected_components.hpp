#pragma once
#include <CPU_adj_list/CPU_adj_list.hpp>
#include <CPU_adj_list/ThreadPool.h>
#include <queue>
#include <vector>

// Weakly Connected Components Algorithm
// call this function like: ans_cpu = CPU_WCC(graph);
// used to show the weakly connected components (ignore the direction of edges) in the graph
// return the component number of each node
template<typename T> // T is float or double
std::vector<int> CPU_connected_components(std::vector<std::vector<std::pair<int, T>>>& input_graph, std::vector<std::vector<std::pair<int, T>>>& output_graph) {
	int N = input_graph.size(); // number of vertices in the graph

	std::vector<int> component(N); // a disjoint-set, stores the component of each vertex
	for (int u = 0; u < N; u++) {
		component[u] = u; // the initial component of each vertex
	}

	bool change = true; 
	size_t threads = std::thread::hardware_concurrency(); // detect the max available hardware concurrency
	std::vector<std::future<void>> results_dynamic(threads); // use to call async codes and synchronize the status of future

	ThreadPool pool_dynamic(threads); // use to initialize the thread pool

	while (change) {
		change = false; // if there is no change in an iteration, change will be set to false
		for (long long q = 0; q < threads; q++) {
			results_dynamic[q] = (pool_dynamic.enqueue([q, N, threads, &change, &input_graph, &component]
			{	// distribute the tasks to each thread
				int start = q * N / threads, end = std::min(N - 1, (int)((q + 1) * N / threads)); // each task computes the subgraph
				for (int u = start; u <= end; u++) {
					for (auto& x : input_graph[u]) {
						int v = x.first;
						int comp_u = component[u];
						int comp_v = component[v]; // get the component of u & v
						if (comp_u == comp_v) continue; // the same component, no need to change
						int high_comp = comp_u > comp_v ? comp_u : comp_v;
						int low_comp = comp_u + (comp_v - high_comp);
						if (high_comp == component[high_comp]) { // if the vertex with high_comp is the root of its disjoint-set, showing that it hasn't been merged, we should merge it into low_comp
							change = true; // the loop should go on
							component[high_comp] = low_comp; // merge
						}
					}
				}
			}));
		}
		for (auto&& result : results_dynamic) {
            result.wait(); // block the process to synchronize the status
        }
		for (long long q = 0; q < threads; q++) { // dsu path compression
			results_dynamic[q] = (pool_dynamic.enqueue([q, N, threads, &component]
			{
				int start = q * N / threads, end = std::min(N - 1, (int)((q + 1) * N / threads));
				for (int u = start; u <= end; u++) {
					while (component[u] != component[component[u]]) { // recursively update the root node of node u to a higher-level ancestor
						component[u] = component[component[u]];
					}
				}
			}));
		}
		for (auto&& result : results_dynamic) {
            result.wait(); // block the process to synchronize the status
        }
	}

	return component;
}

std::vector<std::pair<std::string, std::string>> CPU_WCC(graph_structure<double> & graph){
	std::vector<int> wccVec = CPU_connected_components(graph.OUTs, graph.INs);
	return graph.res_trans_id_id(wccVec);
}
