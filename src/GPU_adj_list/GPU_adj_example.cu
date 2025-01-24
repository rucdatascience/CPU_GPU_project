#include <string>

#include <GPU_adj_list/GPU_adj.hpp>

#include <GPU_adj_list/algorithm/GPU_BFS_adj.cuh>
#include <GPU_adj_list/algorithm/GPU_WCC_adj.cuh>
#include <GPU_adj_list/algorithm/GPU_SSSP_adj.cuh>
#include <GPU_adj_list/algorithm/GPU_PageRank_adj.cuh>
#include <GPU_adj_list/algorithm/GPU_CDLP_adj.cuh>

int main() {
    ios::sync_with_stdio(false);
    std::cin.tie(0), std::cout.tie(0);

    graph_structure<double> graph; // directed graph

    // Add vertices and edges
    graph.add_vertice("one");
    graph.add_vertice("two");
    graph.add_vertice("three");
    graph.add_vertice("four");
    graph.add_vertice("five");
    graph.add_vertice("R");

    GPU_adj<double> gpu_adj = to_GPU_adj(graph);

    graph.add_edge("one", "two", 0.8);
    graph.add_edge("two", "three", 1);
    graph.add_edge("two", "R", 1);
    graph.add_edge("two", "four", 0.1);
    graph.add_edge("R", "three", 1);
    graph.add_edge("one", "three", 1);
    graph.add_edge("one", "four", 1);
    graph.add_edge("four", "three", 1);
    graph.add_edge("four", "five", 1);

    gpu_adj.add_edge(graph.vertex_str_to_id["one"], graph.vertex_str_to_id["two"], 0.8);
    gpu_adj.add_edge(graph.vertex_str_to_id["two"], graph.vertex_str_to_id["three"], 1);
    gpu_adj.add_edge(graph.vertex_str_to_id["two"], graph.vertex_str_to_id["R"], 1);
    gpu_adj.add_edge(graph.vertex_str_to_id["two"], graph.vertex_str_to_id["four"], 0.1);
    gpu_adj.add_edge(graph.vertex_str_to_id["R"], graph.vertex_str_to_id["three"], 1);
    gpu_adj.add_edge(graph.vertex_str_to_id["one"], graph.vertex_str_to_id["three"], 1);
    gpu_adj.add_edge(graph.vertex_str_to_id["one"], graph.vertex_str_to_id["four"], 1);
    gpu_adj.add_edge(graph.vertex_str_to_id["four"], graph.vertex_str_to_id["three"], 1);
    gpu_adj.add_edge(graph.vertex_str_to_id["four"], graph.vertex_str_to_id["five"], 1);

    // Remove a vertex
    gpu_adj.remove_vertex(graph.vertex_str_to_id["R"]); // Remove vertex R by GPU_adj first
    graph.remove_vertice("R");

    // Add a vertex
    graph.add_vertice("six");
    gpu_adj.add_vertex(graph.vertex_str_to_id["six"]);

    // Remove an edge
    graph.remove_edge("two", "four");
    gpu_adj.remove_edge(graph.vertex_str_to_id["two"], graph.vertex_str_to_id["four"]);

    // Add an edge
    graph.add_edge("one", "six", 1);
    gpu_adj.add_edge(graph.vertex_str_to_id["one"], graph.vertex_str_to_id["six"], 1);


    // BFS
    std::cout << "Running BFS..." << std::endl;
    std::vector<std::pair<std::string, int>> gpu_bfs_res = Cuda_Bfs_adj(graph, gpu_adj, "one");
    std::cout << "BFS result: " << std::endl;
    for (auto& res : gpu_bfs_res)
        std::cout << res.first << " " << res.second << std::endl;

    // Connected Components
    std::cout << "Running Connected Components..." << std::endl;
    std::vector<std::pair<std::string, std::string>> gpu_wcc_res = Cuda_WCC_adj(graph, gpu_adj);
    std::cout << "Connected Components result: " << std::endl;
    for (auto& res : gpu_wcc_res)
        std::cout << res.first << " " << res.second << std::endl;

    // SSSP
    std::cout << "Running SSSP..." << std::endl;
    std::vector<std::pair<std::string, double>> gpu_sssp_res = Cuda_SSSP_adj(graph, gpu_adj, "one");
    std::cout << "SSSP result: " << std::endl;
    for (auto& res : gpu_sssp_res)
        std::cout << res.first << " " << res.second << std::endl;

    // PageRank
    std::cout << "Running PageRank..." << std::endl;
    std::vector<std::pair<std::string, double>> gpu_pr_res = Cuda_PR_adj(graph, gpu_adj, 10, 0.85);
    std::cout << "PageRank result: " << std::endl;
    for (auto& res : gpu_pr_res)
        std::cout << res.first << " " << res.second << std::endl;

    // Community Detection
    std::cout << "Running Community Detection..." << std::endl;
    std::vector<std::pair<std::string, std::string>> gpu_cd_res = Cuda_CDLP_adj(graph, gpu_adj, 10);
    std::cout << "Community Detection result: " << std::endl;
    for (auto& res : gpu_cd_res)
        std::cout << res.first << " " << res.second << std::endl;

    return 0;
}
