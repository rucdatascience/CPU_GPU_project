#include <string>

#include <GPU_csr/GPU_csr.hpp>
#include <GPU_csr/algorithm/GPU_BFS.cuh>
#include <GPU_csr/algorithm/GPU_connected_components.cuh>
#include <GPU_csr/algorithm/GPU_shortest_paths.cuh>
#include <GPU_csr/algorithm/GPU_PageRank.cuh>
#include <GPU_csr/algorithm/GPU_Community_Detection.cuh>

int main() {
    ios::sync_with_stdio(false);
    std::cin.tie(0), std::cout.tie(0);

    graph_structure<double> graph; // directed graph

    std::cout << "Input the vertex, type -1 to end" << std::endl;
    while (true) {
        std::string vertex;
        std::cin >> vertex;
        if (vertex == "-1")
            break;
        graph.add_vertice(vertex); // Add a vertex to the graph
    }

    std::cout << "Input the edge, type -1 to end" << std::endl;
    while (true) {
        std::string line_content;
        getline(std::cin, line_content);
        if (line_content == "-1")
            break;
        std::vector<std::string> Parsed_content = parse_string(line_content, " ");
        if (Parsed_content.size() != 3) {
            std::cerr << "Invalid edge input!" << std::endl;
            std::cerr << line_content << std::endl;
            continue;
        }
        graph.add_edge(Parsed_content[0], Parsed_content[1], std::stod(Parsed_content[2])); // Add an edge to the graph
    }

    std::cout << "Remove a vertex, type -1 to end" << std::endl;
    while (true) {
        std::string vertex;
        std::cin >> vertex;
        if (vertex == "-1")
            break;
        graph.remove_vertice(vertex); // Remove a vertex from the graph
    }

    std::cout << "Add a vertex, type -1 to end" << std::endl;
    while (true) {
        std::string vertex;
        std::cin >> vertex;
        if (vertex == "-1")
            break;
        graph.add_vertice(vertex); // Add a vertex to the graph
    }

    std::cout << "Remove an edge, type -1 to end" << std::endl;
    while (true) {
        std::string line_content;
        getline(std::cin, line_content);
        if (line_content == "-1")
            break;
        std::vector<std::string> Parsed_content = parse_string(line_content, " ");
        if (Parsed_content.size() != 2) {
            std::cerr << "Invalid edge input!" << std::endl;
            std::cerr << line_content << std::endl;
            continue;
        }
        graph.remove_edge(Parsed_content[0], Parsed_content[1]); // Remove an edge from the graph
    }

    std::cout << "Add an edge, type -1 to end" << std::endl;
    while (true) {
        std::string line_content;
        getline(std::cin, line_content);
        if (line_content == "-1")
            break;
        std::vector<std::string> Parsed_content = parse_string(line_content, " ");
        if (Parsed_content.size() != 3) {
            std::cerr << "Invalid edge input!" << std::endl;
            continue;
        }
        graph.add_edge(Parsed_content[0], Parsed_content[1], std::stod(Parsed_content[2])); // Add an edge to the graph
    }

    CSR_graph<double> csr_graph = toCSR(graph);

    // BFS
    std::cout << "Input the source vertex for BFS: " << std::endl;
    std::string bfs_src_name;
    std::cin >> bfs_src_name;
    if (graph.vertex_str_to_id.find(bfs_src_name) == graph.vertex_str_to_id.end()) {
        std::cout << "Invalid source vertex for BFS" << std::endl;
        exit(1);
    }
    std::cout << "Running BFS..." << std::endl;
    std::vector<std::pair<std::string, int>> gpu_bfs_res = Cuda_Bfs(graph, csr_graph, bfs_src_name);
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
    std::cout << "Input the source vertex for SSSP: " << std::endl;
    std::string sssp_src_name;
    std::cin >> sssp_src_name;
    if (graph.vertex_str_to_id.find(sssp_src_name) == graph.vertex_str_to_id.end()) {
        std::cout << "Invalid source vertex for SSSP" << std::endl;
        exit(1);
    }
    std::cout << "Running SSSP..." << std::endl;
    std::vector<std::pair<std::string, double>> gpu_sssp_res = Cuda_SSSP(graph, csr_graph, sssp_src_name);
    std::cout << "SSSP result: " << std::endl;
    for (auto& res : gpu_sssp_res)
        std::cout << res.first << " " << res.second << std::endl;

    // PageRank
    std::cout << "Input the iteration number for PageRank: " << std::endl;
    int iteration;
    std::cin >> iteration;
    std::cout << "Input the damping factor for PageRank: " << std::endl;
    double damping_factor;
    std::cin >> damping_factor;
    std::cout << "Running PageRank..." << std::endl;
    std::vector<std::pair<std::string, double>> gpu_pr_res = Cuda_PR(graph, csr_graph, iteration, damping_factor);
    std::cout << "PageRank result: " << std::endl;
    for (auto& res : gpu_pr_res)
        std::cout << res.first << " " << res.second << std::endl;

    // Community Detection
    std::cout << "Input the maximum iteration number for Community Detection: " << std::endl;
    int max_iteration;
    std::cin >> max_iteration;
    std::cout << "Running Community Detection..." << std::endl;
    std::vector<std::pair<std::string, std::string>> gpu_cd_res = Cuda_CDLP(graph, csr_graph, max_iteration);
    std::cout << "Community Detection result: " << std::endl;
    for (auto& res : gpu_cd_res)
        std::cout << res.first << " " << res.second << std::endl;

    return 0;
}