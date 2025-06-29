#include <string>

#include <GPU_csr/GPU_csr.hpp>

#include <GPU_csr/algorithm/GPU_BFS_csr.cuh>
#include <GPU_csr/algorithm/GPU_WCC_csr.cuh>
#include <GPU_csr/algorithm/GPU_SSSP_csr.cuh>
#include <GPU_csr/algorithm/GPU_SSSP_pre_csr.cuh>
#include <GPU_csr/algorithm/GPU_PR_csr.cuh>
#include <GPU_csr/algorithm/GPU_CDLP_csr.cuh>

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

    graph.add_edge("one", "two", 0.8);
    graph.add_edge("two", "three", 1);
    graph.add_edge("two", "R", 1);
    graph.add_edge("two", "four", 0.1);
    graph.add_edge("R", "three", 1);
    graph.add_edge("one", "three", 1);
    graph.add_edge("one", "four", 1);
    graph.add_edge("four", "three", 1);
    graph.add_edge("four", "five", 1);

    // Remove a vertex
    graph.remove_vertice("R");

    // Add a vertex
    graph.add_vertice("six");

    // Remove an edge
    graph.remove_edge("two", "four");

    // Add an edge
    graph.add_edge("one", "six", 1);

    // Transform to CSR
    auto begin = std::chrono::high_resolution_clock::now();
    CSR_graph<double> csr_graph = toCSR(graph);
    auto end = std::chrono::high_resolution_clock::now();
    double ldbc_to_csr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    printf("LDBC to CSR cost time: %f s\n", ldbc_to_csr_time);
    // return 0;

    // BFS
    std::cout << "Running BFS..." << std::endl;
    std::vector<std::pair<std::string, int>> gpu_bfs_res = Cuda_BFS(graph, csr_graph, "one");
    std::cout << "BFS result: " << std::endl;
    for (auto& res : gpu_bfs_res)
        std::cout << res.first << " " << res.second << std::endl;

    // Connected Components
    std::cout << "Running Connected Components..." << std::endl;
    std::vector<std::pair<std::string, std::string>> gpu_wcc_res = Cuda_WCC(graph, csr_graph);
    std::cout << "Connected Components result: " << std::endl;
    for (auto& res : gpu_wcc_res)
        std::cout << res.first << " " << res.second << std::endl;

    // SSSP
    std::cout << "Running SSSP..." << std::endl;
    std::vector<std::pair<std::string, double>> gpu_sssp_res = Cuda_SSSP(graph, csr_graph, "one", std::numeric_limits<double>::max());
    std::cout << "SSSP result: " << std::endl;
    std::vector<int> pre_v;
    gpu_sssp_res = Cuda_SSSP_pre(graph, csr_graph, "one", pre_v, std::numeric_limits<double>::max());
    for (auto& res : gpu_sssp_res)
        std::cout << res.first << " " << res.second << std::endl;

    // PageRank
    std::cout << "Running PageRank..." << std::endl;
    std::vector<std::pair<std::string, double>> gpu_pr_res = Cuda_PR(graph, csr_graph, 10, 0.85);
    std::cout << "PageRank result: " << std::endl;
    for (auto& res : gpu_pr_res)
        std::cout << res.first << " " << res.second << std::endl;

    // Community Detection
    std::cout << "Running Community Detection..." << std::endl;
    std::vector<std::pair<std::string, std::string>> gpu_cd_res = Cuda_CDLP(graph, csr_graph, 10);
    std::cout << "Community Detection result: " << std::endl;
    for (auto& res : gpu_cd_res)
        std::cout << res.first << " " << res.second << std::endl;

    return 0;
}