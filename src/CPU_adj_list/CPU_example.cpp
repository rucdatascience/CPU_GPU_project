#include <string>

#include <CPU_adj_list/CPU_adj_list.hpp>
#include <CPU_adj_list/algorithm/CPU_BFS.hpp>
#include <CPU_adj_list/algorithm/CPU_connected_components.hpp>
#include <CPU_adj_list/algorithm/CPU_shortest_paths.hpp>
#include <CPU_adj_list/algorithm/CPU_PageRank.hpp>
#include <CPU_adj_list/algorithm/CPU_Community_Detection.hpp>

int main()
{
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

    // BFS
    std::cout << "Input the source vertex for BFS: " << std::endl;
    std::string bfs_src_name;
    std::cin >> bfs_src_name;
    if (graph.vertex_str_to_id.find(bfs_src_name) == graph.vertex_str_to_id.end()) {
        std::cout << "Invalid source vertex for BFS" << std::endl;
        exit(1);
    }
    std::cout << "Running BFS..." << std::endl;
    std::vector<std::pair<std::string, int>> cpu_bfs_result = CPU_Bfs(graph, bfs_src_name);
    std::cout << "BFS result: " << std::endl;
    for (auto& res : cpu_bfs_result)
        std::cout << res.first << " " << res.second << std::endl;

    // Connected Components
    std::cout << "Running Connected Components..." << std::endl;
    std::vector<std::pair<std::string, std::string>> cpu_connected_components_result = CPU_WCC(graph);
    std::cout << "Connected Components result: " << std::endl;
    for (auto& res : cpu_connected_components_result)
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
    std::vector<std::pair<std::string, double>> cpu_sssp_result = CPU_SSSP(graph, sssp_src_name);
    std::cout << "SSSP result: " << std::endl;
    for (auto& res : cpu_sssp_result)
        std::cout << res.first << " " << res.second << std::endl;

    // PageRank
    std::cout << "Input the iteration number for PageRank: " << std::endl;
    int iteration;
    std::cin >> iteration;
    std::cout << "Input the damping factor for PageRank: " << std::endl;
    double damping_factor;
    std::cin >> damping_factor;
    std::cout << "Running PageRank..." << std::endl;
    std::vector<std::pair<std::string, double>> cpu_pagerank_result = CPU_PR(graph, iteration, damping_factor);
    std::cout << "PageRank result: " << std::endl;
    for (auto& res : cpu_pagerank_result)
        std::cout << res.first << " " << res.second << std::endl;

    // Community Detection
    std::cout << "Input the maximum iteration number for Community Detection: " << std::endl;
    int max_iteration;
    std::cin >> max_iteration;
    std::cout << "Running Community Detection..." << std::endl;
    std::vector<std::pair<std::string, std::string>> cpu_community_detection_result = CPU_CDLP(graph, max_iteration);
    std::cout << "Community Detection result: " << std::endl;
    for (auto& res : cpu_community_detection_result)
        std::cout << res.first << " " << res.second << std::endl;

    return 0;
}