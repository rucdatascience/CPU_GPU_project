#pragma once

/*the following codes are for testing

---------------------------------------------------
a cpp file (try.cpp) for running the following test code:
----------------------------------------

#include <iostream>
#include <fstream>
using namespace std;

// header files in the Boost library: https://www.boost.org/
#include <boost/random.hpp>
boost::random::mt19937 boost_random_time_seed{ static_cast<std::uint32_t>(std::time(0)) };

#include <build_in_progress/CPU_GPU_project/test_CPU.h>


int main()
{
	test_CPU();
}

------------------------------------------------------------------------------------------
Commends for running the above cpp file on Linux:

g++ -std=c++17 -I/home/boost_1_75_0 -I/root/rucgraph try.cpp -lpthread -Ofast -o A
./A
rm A

(optional to put the above commends in run.sh, and then use the comment: sh run.sh)


*/
#include <chrono>
#include <graph_v_of_v/graph_v_of_v.h>
#include <build_in_progress/CPU_GPU_project/CPU_BFS.h>
#include <build_in_progress/CPU_GPU_project/CPU_shortest_paths.h>
#include <build_in_progress/CPU_GPU_project/CPU_connected_components.h>
#include <build_in_progress/CPU_GPU_project/CPU_PageRank.h>
#include <build_in_progress/CPU_GPU_project/CPU_Community_Detection.h>



void test_CPU() {

	graph_v_of_v<int> instance_graph;
	instance_graph.txt_read("example_graph.txt"); // need to put example_graph.txt into the default path

	/*BFS*/
	if (1) {
		auto begin = std::chrono::high_resolution_clock::now();
		
		int N = instance_graph.size();
		for (int v = 0; v < N; v++) {
			CPU_BFS<int>(instance_graph, v);
		}

		double CPU_time_connected_components = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() / 1e9; // s
		cout << "CPU_BFS / N: " << CPU_time_connected_components / N << "s" << endl;
	}

	/*connected_components*/
	if (1) {
		auto begin = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < 100; i++) {
			CPU_connected_components<int>(instance_graph);
		}

		double CPU_time_connected_components = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() / 1e9; // s
		cout << "CPU_time_connected_components: " << CPU_time_connected_components / 100 << "s" << endl;
	}

	/*shortest_paths*/
	if (1) {
		auto begin = std::chrono::high_resolution_clock::now();

		std::vector<int> distances, predecessors;
		int N = instance_graph.size();
		for (int v = 0; v < N; v++) {
			CPU_shortest_paths<int>(instance_graph, v, distances, predecessors);
		}
		
		double CPU_time_shortest_paths = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() / 1e9; // s
		cout << "CPU_time_shortest_paths / N: " << CPU_time_shortest_paths / N << "s" << endl;
	}


	/*PageRank*/
	if (1) {
		auto begin = std::chrono::high_resolution_clock::now();

		/*PageRank function*/

		double CPU_time_PageRank = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() / 1e9; // s
		cout << "CPU_time_PageRank: " << CPU_time_PageRank << "s" << endl;
	}


	/*Community_Detection*/
	if (1) {
		auto begin = std::chrono::high_resolution_clock::now();

		/*Community_Detection function*/

		double CPU_time_Community_Detection = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() / 1e9; // s
		cout << "CPU_time_Community_Detection: " << CPU_time_Community_Detection << "s" << endl;
	}



}









