#include <string>

#include <GPU_gpma/GPU_gpma.hpp>

#include <GPU_gpma/algorithm/GPU_BFS_gpma_optimized.cuh>
#include <GPU_gpma/algorithm/GPU_WCC_gpma_optimized.cuh>
#include <GPU_gpma/algorithm/GPU_SSSP_gpma_optimized.cuh>
#include <GPU_gpma/algorithm/GPU_PR_gpma_optimized.cuh>
#include <GPU_gpma/algorithm/GPU_CDLP_gpma_optimized.cuh>

#include <GPU_gpma/algorithm/GPU_BFS_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_WCC_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_SSSP_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_PR_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_CDLP_gpma.cuh>

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

    // 0 1
    // 1 2
    // 1 3
    // 0 2
    // 0 3
    // 3 2
    // 3 4
    graph.add_edge("one", "two", 0.81111111111111111111111111);
    graph.add_edge("two", "three", 1.111111111111111111111111111111);
    graph.add_edge("two", "R", 1.1111111111111111111111111);
    graph.add_edge("two", "four", 0.1);
    graph.add_edge("R", "three", 1.11111111111111111111);
    graph.add_edge("one", "three", 1.1111111111111111111111111111);
    graph.add_edge("one", "four", 1.1111111111111111111111111);
    graph.add_edge("four", "three", 1.1111111111111111111111111);
    graph.add_edge("four", "five", 1.1111111111111111111111111);
    
    // Remove a vertex
    graph.remove_vertice("R");

    // Add a vertex
    graph.add_vertice("six");

    // Remove an edge
    graph.remove_edge("two", "four");

    // Add an edge
    // graph.add_edge("one", "six", 1);
    
    // add
    // graph.add_edge("three", "five", 1);
    // graph.add_edge("five", "two", 1);
    // graph.add_edge("six", "three", 1);
    
    // remove
    graph.remove_edge("four", "five");
    graph.remove_edge("one", "two");
    graph.remove_edge("one", "three");
    graph.remove_edge("one", "four");
    
    // Transform to GPMA
    auto begin = std::chrono::high_resolution_clock::now();

    // Step 1: CPU adjacency list
    int num_nodes = graph.size();
    std::vector<std::vector<std::pair<KEY_TYPE, VALUE_TYPE>>> adj_list_out(num_nodes), adj_list_in(num_nodes);
    for (int v = 0; v < num_nodes; ++ v) {
        for (auto it = graph.OUTs[v].begin(); it != graph.OUTs[v].end(); it ++) {
            // std::cout << "outs: " << v <<", " << it->first << std::endl;
            adj_list_out[v].push_back({static_cast<KEY_TYPE>(it->first), static_cast<VALUE_TYPE>(it->second)});
        }
        for (auto it = graph.INs[v].begin(); it != graph.INs[v].end(); it ++) {
            // std::cout << "ins: " << v <<", " << it->first << std::endl;
            adj_list_in[v].push_back({static_cast<KEY_TYPE>(it->first), static_cast<VALUE_TYPE>(it->second)});
        }
    }
    
    // Step 2: Constructing key-value pairs and row offsets
    std::vector<KEY_TYPE> cpu_keys_out, cpu_keys_in;
    std::vector<VALUE_TYPE> cpu_values_out, cpu_values_in;
    std::vector<SIZE_TYPE> cpu_row_offset_out(num_nodes + 1, 0), cpu_row_offset_in(num_nodes + 1, 0);
    SIZE_TYPE current_pos_out = 0, current_pos_in = 0;

    for (KEY_TYPE u = 0; u < num_nodes; ++u) {
        cpu_row_offset_out[u] = current_pos_out, cpu_row_offset_in[u] = current_pos_in;
        for (const auto& edge : adj_list_out[u]) {
            KEY_TYPE v = edge.first;
            VALUE_TYPE weight = edge.second;
            KEY_TYPE key = (u << 32) | v;
            VALUE_TYPE value = static_cast<VALUE_TYPE>(weight);
            cpu_keys_out.push_back(key);
            cpu_values_out.push_back(value);
            current_pos_out ++;
        }
        for (const auto& edge : adj_list_in[u]) {
            KEY_TYPE v = edge.first;
            VALUE_TYPE weight = edge.second;
            KEY_TYPE key = (u << 32) | v;
            VALUE_TYPE value = static_cast<VALUE_TYPE>(weight);
            cpu_keys_in.push_back(key);
            cpu_values_in.push_back(value);
            current_pos_in ++;
        }
    }
    cpu_row_offset_out[num_nodes] = current_pos_out, cpu_row_offset_in[num_nodes] = current_pos_in;

    // Step 3: init GPMA_out, GPMA_in
    GPMA gpma_graph_out, gpma_graph_in;
    SIZE_TYPE row_num = num_nodes;
    init_csr_gpma(gpma_graph_out, row_num);
    init_csr_gpma(gpma_graph_in, row_num);

    // Step 4: Copy data to the device
    DEV_VEC_KEY dev_keys_out(cpu_keys_out.size()), dev_keys_in(cpu_keys_in.size());
    DEV_VEC_VALUE dev_values_out(cpu_keys_out.size()), dev_values_in(cpu_keys_in.size());
    DEV_VEC_SIZE dev_row_offset_out(cpu_row_offset_out.size()), dev_row_offset_in(cpu_row_offset_in.size());

    thrust::copy(cpu_keys_out.begin(), cpu_keys_out.end(), dev_keys_out.begin());
    thrust::copy(cpu_keys_in.begin(), cpu_keys_in.end(), dev_keys_in.begin());
    thrust::copy(cpu_values_out.begin(), cpu_values_out.end(), dev_values_out.begin());
    thrust::copy(cpu_values_in.begin(), cpu_values_in.end(), dev_values_in.begin());
    thrust::copy(cpu_row_offset_out.begin(), cpu_row_offset_out.end(), dev_row_offset_out.begin());
    thrust::copy(cpu_row_offset_in.begin(), cpu_row_offset_in.end(), dev_row_offset_in.begin());
    cudaDeviceSynchronize();
    gpma_graph_out.row_offset = dev_row_offset_out;
    gpma_graph_in.row_offset = dev_row_offset_in;
    
    // Step 5: Insert data into GPMA
    init_gpma_from_csr(gpma_graph_out, dev_keys_out, dev_values_out);
    init_gpma_from_csr(gpma_graph_in, dev_keys_in, dev_values_in);
    // update_gpma(gpma_graph_out, dev_keys_out, dev_values_out);
    // update_gpma(gpma_graph_in, dev_keys_in, dev_values_in);

    auto end = std::chrono::high_resolution_clock::now();
    double ldbc_to_gpma_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    printf("LDBC to GPMA cost time: %f s\n", ldbc_to_gpma_time);
    
    // Allocation result memory (device)
    DEV_VEC_VALUE results(num_nodes, 0);

    // Processing result (such as copying back to the host)
    std::vector<VALUE_TYPE> cpu_results(num_nodes);
    cudaMemcpy(cpu_results.data(), RAW_PTR(results), num_nodes * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // BFS
    std::cout << "Running BFS..." << std::endl;
    std::vector<std::pair<std::string, int>> gpu_bfs_res = Cuda_BFS_optimized(graph, gpma_graph_out, "one");
    std::cout << "BFS result: " << std::endl;
    for (auto& res : gpu_bfs_res)
        std::cout << res.first << " " << res.second << std::endl;

    // SSSP
    std::cout << "Running SSSP..." << std::endl;
    std::vector<std::pair<std::string, double>> gpu_sssp_res = Cuda_SSSP_optimized(graph, gpma_graph_out, "one");
    std::cout << "SSSP result: " << std::endl;
    for (auto& res : gpu_sssp_res)
        std::cout << res.first << " " << res.second << std::endl;

    // Connected Components
    std::cout << "Running Connected Components..." << std::endl;
    std::vector<std::pair<std::string, std::string>> gpu_wcc_res = Cuda_WCC_optimized(graph, gpma_graph_in);
    std::cout << "Connected Components result: " << std::endl;
    for (auto& res : gpu_wcc_res)
        std::cout << res.first << " " << res.second << std::endl;

    // PageRank
    std::cout << "Running PageRank..." << std::endl;
    std::vector<std::pair<std::string, double>> gpu_pr_res_v2 = Cuda_PR_optimized(graph, gpma_graph_in, gpma_graph_out, 10, 0.85);
    std::cout << "PageRank result: " << std::endl;
    for (auto& res : gpu_pr_res_v2)
        std::cout << res.first << " " << res.second << std::endl;
    
    // Community Detection
    std::cout << "Running Community Detection..." << std::endl;
    std::vector<std::pair<std::string, std::string>> gpu_cd_res = Cuda_CDLP_optimized(graph, gpma_graph_in, gpma_graph_out, 10);
    std::cout << "Community Detection result: " << std::endl;
    for (auto& res : gpu_cd_res)
        std::cout << res.first << " " << res.second << std::endl;

    return 0;
}