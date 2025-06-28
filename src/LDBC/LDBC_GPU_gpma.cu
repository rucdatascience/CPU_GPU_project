#include <time.h>
#include <chrono>
#include <sstream>
#include <iostream>

#include <GPU_gpma/algorithm/GPU_BFS_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_WCC_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_SSSP_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_PR_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_CDLP_gpma.cuh>

#include <GPU_gpma/algorithm/GPU_BFS_gpma_optimized.cuh>
#include <GPU_gpma/algorithm/GPU_WCC_gpma_optimized.cuh>
#include <GPU_gpma/algorithm/GPU_SSSP_gpma_optimized.cuh>
#include <GPU_gpma/algorithm/GPU_PR_gpma_optimized.cuh>
#include <GPU_gpma/algorithm/GPU_CDLP_gpma_optimized.cuh>

#include <LDBC/checker.hpp>
#include <LDBC/ldbc.hpp>

std::vector<std::pair<std::string, std::string> > vec_add;
std::vector<std::pair<std::string, std::string> > vec_delete;

inline void read_edge_file (std::string graph_file_path) {
    std::ifstream infile_add(graph_file_path + "-add-edge.txt");
    std::ifstream infile_delete(graph_file_path + "-delete-edge.txt");

    // read the file
    std::string line;
    while (std::getline(infile_add, line)) {
        std::istringstream iss(line);
        std::string c, a, b;
        iss >> c >> a >> b;
        vec_add.push_back(std::make_pair(a, b));
    }
    infile_add.close();
    
    while (std::getline(infile_delete, line)) {
        std::istringstream iss(line);
        std::string c, a, b;
        iss >> c >> a >> b;
        vec_delete.push_back(std::make_pair(a, b));
    }
    infile_delete.close();

    return;
}

inline KEY_TYPE make_edge_key(KEY_TYPE src, KEY_TYPE dst) {
    return ((KEY_TYPE)src << 32) | (dst & 0xFFFFFFFF);
}

inline void batch_add_edge(GPMA& gpma_in, GPMA& gpma_out, graph_structure<double>& graph) {
    std::string e1, e2;
    size_t n = vec_add.size();
    DEV_VEC_KEY keys_in(n), keys_out(n);
    DEV_VEC_VALUE values_in(n), values_out(n);
    
    for (size_t i = 0; i < n; ++i) {
        e1 = vec_add[i].first, vec_add[i].second;
        int v1 = graph.vertex_str_to_id[e1], v2 = graph.vertex_str_to_id[e2];
        keys_in[i] = make_edge_key(v1, v2);
        keys_out[i] = make_edge_key(v2, v1);
        values_in[i] = 1, values_out[i] = 1;
    }

    update_gpma(gpma_in, keys_in, values_in), update_gpma(gpma_out, keys_out, values_out);
}

inline void batch_delete_edge(GPMA& gpma_in, GPMA& gpma_out, graph_structure<double>& graph) {
    std::string e1, e2;
    size_t n = vec_add.size();
    DEV_VEC_KEY keys_in(n), keys_out(n);
    DEV_VEC_VALUE values_in(n), values_out(n);
    
    for (size_t i = 0; i < n; ++i) {
        e1 = vec_delete[i].first, vec_delete[i].second;
        int v1 = graph.vertex_str_to_id[e1], v2 = graph.vertex_str_to_id[e2];
        keys_in[i] = make_edge_key(v1, v2);
        keys_out[i] = make_edge_key(v2, v1);
        values_in[i] = VALUE_NONE, values_out[i] = VALUE_NONE;
    }

    update_gpma(gpma_in, keys_in, values_in), update_gpma(gpma_out, keys_out, values_out);
}

inline void add_edge(GPMA& gpma_in, GPMA& gpma_out, graph_structure<double>& graph) {
    std::string e1, e2;
    for (int i = 0; i < vec_add.size(); i ++) {
        e1 = vec_add[i].first, vec_add[i].second;
        int v1 = graph.vertex_str_to_id[e1], v2 = graph.vertex_str_to_id[e2];
        
        // Pack in the form of key-value pairs
        DEV_VEC_KEY keys_in(1, make_edge_key(v2, v1)), keys_out(1, make_edge_key(v1, v2));
        DEV_VEC_VALUE values_in(1, 1), values_out(1, 1);

        // Call the function "update_gpma" to update the graph structure
        update_gpma(gpma_in, keys_in, values_in), update_gpma(gpma_out, keys_out, values_out);
    }
}

inline void delete_edge(GPMA& gpma_in, GPMA& gpma_out, graph_structure<double>& graph) {
    std::string e1, e2;
    for (int i = 0; i < vec_delete.size(); i ++) {
        e1 = vec_delete[i].first, e2 = vec_delete[i].second;
        int v1 = graph.vertex_str_to_id[e1], v2 = graph.vertex_str_to_id[e2];
    
        // Pack in the form of key-value pairs
        DEV_VEC_KEY keys_in(1, make_edge_key(v2, v1)), keys_out(1, make_edge_key(v1, v2));
        DEV_VEC_VALUE values_in(1, VALUE_NONE), values_out(1, VALUE_NONE);

        // Call the function "update_gpma" to update the graph structure
        update_gpma(gpma_in, keys_in, values_in), update_gpma(gpma_out, keys_out, values_out);
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);

    //freopen("../input.txt", "r", stdin);

    std::vector<std::pair<std::string, std::string>> result_all;

    std::string directory;
    std::cout << "Please input the data directory: " << std::endl;
    std::cin >> directory;

    if (directory.back() != '/')
        directory += "/";

    std::string graph_name;
    std::cout << "Please input the graph name: " << std::endl;
    std::cin >> graph_name;

    std::string config_file_path = directory + graph_name + ".properties";

    LDBC<double> graph(directory, graph_name);
    graph.read_config(config_file_path); //Read the ldbc configuration file to obtain key parameter information in the file

    auto begin = std::chrono::high_resolution_clock::now();
    graph.load_graph(); //Read the vertex and edge files corresponding to the configuration file, // The vertex information in graph is converted to csr format for storage   
    auto end = std::chrono::high_resolution_clock::now();
    double load_ldbc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    printf("load_ldbc_time cost time: %f s\n", load_ldbc_time);

    begin = std::chrono::high_resolution_clock::now();
    // Step 1: CPU adjacency list
    int num_nodes = graph.size();
    std::vector<std::vector<std::pair<KEY_TYPE, VALUE_TYPE>>> adj_list_out(num_nodes), adj_list_in(num_nodes);
    for (int v = 0; v < num_nodes; ++ v) {
        for (auto it = graph.OUTs[v].begin(); it != graph.OUTs[v].end(); it ++) {
            adj_list_out[v].push_back({static_cast<KEY_TYPE>(it->first), static_cast<VALUE_TYPE>(it->second)});
        }
        for (auto it = graph.INs[v].begin(); it != graph.INs[v].end(); it ++) {
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
        // cpu_keys_out.push_back(KEY_NONE & (u << 32));
        // cpu_values_out.push_back(VALUE_NONE);
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
    cudaDeviceSynchronize();

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
    
    dev_keys_out.clear();
    dev_keys_out.shrink_to_fit();
    dev_values_out.clear();
    dev_values_out.shrink_to_fit();
    dev_row_offset_out.clear();
    dev_row_offset_out.shrink_to_fit();

    if (!graph.is_directed) {
        gpma_graph_in = gpma_graph_out;
        dev_keys_in.clear(); dev_keys_in.shrink_to_fit();
        dev_values_in.clear(); dev_values_in.shrink_to_fit();
        dev_row_offset_in.clear(); dev_row_offset_in.shrink_to_fit();
    } else {
        init_gpma_from_csr(gpma_graph_in, dev_keys_in, dev_values_in);
        dev_keys_in.clear(); dev_keys_in.shrink_to_fit();
        dev_values_in.clear(); dev_values_in.shrink_to_fit();
        dev_row_offset_in.clear(); dev_row_offset_in.shrink_to_fit();
    }

    // update_gpma(gpma_graph_out, dev_keys_out, dev_values_out);
    // update_gpma(gpma_graph_in, dev_keys_in, dev_values_in);
    
    end = std::chrono::high_resolution_clock::now();
    double ldbc_to_gpma_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    printf("LDBC to GPMA cost time: %f s\n", ldbc_to_gpma_time);

    // size_t free_mem, total_mem;
    // cudaMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Free VRAM: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    
    // read .e and make data for add edges and delete edges
    // read_edge_file("/home/mdnd/CPU_GPU_project-main/data/" + graph_name);

    // std::vector<std::pair<std::string, std::string>> vector_edge;
    // begin = std::chrono::high_resolution_clock::now();
    // batch_add_edge(gpma_graph_in, gpma_graph_out, graph);
    // end =std::chrono::high_resolution_clock::now();
    // double add_edge_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    // vector_edge.push_back(std::make_pair("add-edge", std::to_string(add_edge_time)));
    // printf("add_edge_time cost time: %f s\n", add_edge_time);
    
    // std::vector<std::pair<std::string, std::string>> vector_delete_edge;
    // begin = std::chrono::high_resolution_clock::now();
    // batch_delete_edge(gpma_graph_in, gpma_graph_out, graph);
    // end =std::chrono::high_resolution_clock::now();
    // double delete_edge_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    // vector_edge.push_back(std::make_pair("delete-edge", std::to_string(delete_edge_time)));
    // printf("delete_edge_time cost time: %f s\n", delete_edge_time);
    
    // graph.save_to_CSV(vector_edge, "./result-gpu-gpma-edge-batch.csv");
    // return 0;

    int iter = 1;
    if (1) {
        if (graph.sup_bfs) {
            double gpu_bfs_time = 0;
            try {
                std::vector<std::pair<std::string, int>> bfs_result;
                begin = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iter; i ++) bfs_result = Cuda_BFS(graph, gpma_graph_out, graph.bfs_src_name);
                end = std::chrono::high_resolution_clock::now();
                gpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU BFS cost time: %f s\n", gpu_bfs_time / (double)iter);
                
                if (Bfs_checker(graph, bfs_result, graph.base_path + "-BFS"))
                    result_all.push_back(std::make_pair("BFS", std::to_string(gpu_bfs_time / (double)iter)));
                else
                    result_all.push_back(std::make_pair("BFS", "wrong"));
            }
            catch (...) {
                result_all.push_back(std::make_pair("BFS", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("BFS", "N/A"));
    }

    if (1) {
        if (graph.sup_sssp) {
            double gpu_sssp_time = 0;
            try {
                std::vector<std::pair<std::string, double>> sssp_result;
                //std::vector<int> pre_v;
                begin = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iter; i ++) sssp_result = Cuda_SSSP(graph, gpma_graph_out, graph.sssp_src_name, std::numeric_limits<double>::max());
                //sssp_result = Cuda_SSSP_pre(graph, csr_graph, graph.sssp_src_name, pre_v, std::numeric_limits<double>::max());
                end = std::chrono::high_resolution_clock::now();
                gpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU SSSP cost time: %f s\n", gpu_sssp_time / (double)iter);
                if (SSSP_checker(graph, sssp_result, graph.base_path + "-SSSP"))
                    result_all.push_back(std::make_pair("SSSP", std::to_string(gpu_sssp_time / (double)iter)));
                else
                    result_all.push_back(std::make_pair("SSSP", "wrong"));
                
                /*std::vector<std::pair<std::string, std::string>> path = path_query(graph, graph.sssp_src_name, "338", pre_v);
                for (auto p : path) {
                    std::cout << p.first << "->" << p.second << std::endl;
                }*/
            }
            catch (...) {
                result_all.push_back(std::make_pair("SSSP", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("SSSP", "N/A"));
    }
    
    if (1) {
        if (graph.sup_wcc) {
            double gpu_wcc_time = 0;
            try {
                std::vector<std::pair<std::string, std::string>> wcc_result;
                begin = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iter; i ++) wcc_result = Cuda_WCC(graph, gpma_graph_out);
                end = std::chrono::high_resolution_clock::now();
                gpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU WCC cost time: %f s\n", gpu_wcc_time / (double)iter);
                if (WCC_checker(graph, wcc_result, graph.base_path + "-WCC"))
                    result_all.push_back(std::make_pair("WCC", std::to_string(gpu_wcc_time / (double)iter)));
                else
                    result_all.push_back(std::make_pair("WCC", "wrong"));
            }
            catch (...) {
                result_all.push_back(std::make_pair("WCC", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("WCC", "N/A"));
    }

    if (1) {
        if (graph.sup_pr) {
            double gpu_pr_time = 0;
            try {
                std::vector<std::pair<std::string, double>> pr_result;
                begin = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iter; i ++) pr_result = Cuda_PR(graph, gpma_graph_in, gpma_graph_out, graph.pr_its, graph.pr_damping);
                end = std::chrono::high_resolution_clock::now();
                gpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU PageRank cost time: %f s\n", gpu_pr_time / (double)iter);
                if (PR_checker(graph, pr_result, graph.base_path + "-PR"))
                    result_all.push_back(std::make_pair("PR", std::to_string(gpu_pr_time / (double)iter)));
                else
                    result_all.push_back(std::make_pair("PR", "wrong"));
            }
            catch (...) {
                result_all.push_back(std::make_pair("PR", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("PR", "N/A"));
    }
    
    if (1) {
        if (graph.sup_cdlp) {
            double gpu_cdlp_time = 0;
            try {
                std::vector<std::pair<std::string, std::string>> cdlp_result;
                begin = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iter; i ++) cdlp_result = Cuda_CDLP(graph, gpma_graph_in, gpma_graph_out, graph.cdlp_max_its);
                end = std::chrono::high_resolution_clock::now();
                gpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU Community Detection cost time: %f s\n", gpu_cdlp_time / (double)iter);
                if (CDLP_checker(graph, cdlp_result, graph.base_path + "-CDLP"))
                    result_all.push_back(std::make_pair("CDLP", std::to_string(gpu_cdlp_time / (double)iter)));
                else
                    result_all.push_back(std::make_pair("CDLP", "wrong"));
            }
            catch (...) {
                result_all.push_back(std::make_pair("CDLP", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("CDLP", "N/A"));
    }

    std::cout << "Result: " << std::endl;
    int res_size = result_all.size();
    for (int i = 0; i < res_size; i++) {
        std::cout << result_all[i].second;
        if (i != res_size - 1)
            std::cout << ",";
    }
    std::cout << std::endl;

    graph.save_to_CSV(result_all, "./result-gpu-gpma.csv");

    return 0;
}
