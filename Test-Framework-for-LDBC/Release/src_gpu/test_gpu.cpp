#include <GPU_BFS.cuh>
#include <GPU_connected_components.cuh>
#include <GPU_shortest_paths.cuh>
#include <GPU_PageRank.cuh>
#include "GPU_Community_Detection.cuh"
#include <chrono>
#include <checker.hpp>
#include <time.h>


int main()
{
    ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);    

    vector<string> datas = {"datagen-7_5-fb.properties"};

    freopen("../data/input.txt", "r", stdin);
    
    for (string config_file : datas) {
        std::vector<std::pair<std::string, std::string>> result_all;

        std::string config_file_path;
        std::cout << "Please input the config file path: ";
        std::cin >> config_file_path;

        // graph_structure<double> graph;
        graph_structure<double> graph;
        graph.read_config(config_file_path); //Read the ldbc configuration file to obtain key parameter information in the file

        auto begin = std::chrono::high_resolution_clock::now();
        graph.load_graph(); //Read the vertex and edge files corresponding to the configuration file, // The vertex information in graph is converted to csr format for storage   
        auto end = std::chrono::high_resolution_clock::now();
        double load_ldbc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
        printf("load_ldbc_time cost time: %f s\n", load_ldbc_time);

        begin = std::chrono::high_resolution_clock::now();
        CSR_graph<double> csr_graph = toCSR(graph);
        end = std::chrono::high_resolution_clock::now();
        double graph_to_csr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
        std::cout << "Number of vertices: " << csr_graph.OUTs_Neighbor_start_pointers.size()-1 << std::endl;
        std::cout << "Number of edges: " << csr_graph.OUTs_Edges.size() << std::endl;
        printf("graph_to_csr_time cost time: %f s\n", graph_to_csr_time);

       
        float elapsedTime = 0;
        unordered_map<string, string> umap_all_res;
        size_t lastSlashPos = config_file.find_last_of("/\\");
        size_t lastDotPos = config_file.find_last_of(".");
        string test_file_name = config_file.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);
        umap_all_res.emplace("test_file_name", test_file_name);

        if (graph.sup_bfs) {
            bool bfs_pass = 0;
            std::vector<std::pair<std::string, int>> bfs_result;
            begin = std::chrono::high_resolution_clock::now();
            bfs_result = Cuda_Bfs(graph, csr_graph, graph.bfs_src_name);
            end = std::chrono::high_resolution_clock::now();
            double gpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("GPU BFS cost time: %f s\n", gpu_bfs_time);
            Bfs_checker(graph, bfs_result, bfs_pass);

            result_all.push_back(std::make_pair("BFS", std::to_string(gpu_bfs_time)));
        }
        else
            result_all.push_back(std::make_pair("BFS", "N/A"));

        if (graph.sup_sssp) {
            bool sssp_pass = 0;

            std::vector<std::pair<std::string, double>> sssp_result;
            begin = std::chrono::high_resolution_clock::now();
            sssp_result = Cuda_SSSP(graph, csr_graph, graph.sssp_src_name, std::numeric_limits<double>::max());
            end = std::chrono::high_resolution_clock::now();
            double gpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("GPU SSSP cost time: %f s\n", gpu_sssp_time);
            SSSP_checker(graph, sssp_result, sssp_pass);

            result_all.push_back(std::make_pair("SSSP", std::to_string(gpu_sssp_time)));
        }
        else
            result_all.push_back(std::make_pair("SSSP", "N/A"));

        if (graph.sup_wcc) {
            bool wcc_pass = false;

            std::vector<std::pair<std::string, std::string>> wcc_result;
            begin = std::chrono::high_resolution_clock::now();
            wcc_result = Cuda_WCC(graph, csr_graph);
            end = std::chrono::high_resolution_clock::now();
            double gpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("GPU WCC cost time: %f s\n", gpu_wcc_time);
            WCC_checker(graph, wcc_result, wcc_pass);

            result_all.push_back(std::make_pair("WCC", std::to_string(gpu_wcc_time)));
        }
        else
            result_all.push_back(std::make_pair("WCC", "N/A"));

        if (graph.sup_pr) {
            bool pr_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, double>> pr_result = Cuda_PR(graph, csr_graph, graph.pr_its, graph.pr_damping);
            end = std::chrono::high_resolution_clock::now();
            double gpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("GPU PageRank cost time: %f s\n", gpu_pr_time);
            PR_checker(graph, pr_result, pr_pass);

            result_all.push_back(std::make_pair("PR", std::to_string(gpu_pr_time)));
        }
        else
            result_all.push_back(std::make_pair("PR", "N/A"));

        if (graph.sup_cdlp) {
            bool cdlp_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, std::string>> cdlp_result = Cuda_CDLP(graph, csr_graph, graph.cdlp_max_its);
            end = std::chrono::high_resolution_clock::now();
            double gpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("GPU Community Detection cost time: %f s\n", gpu_cdlp_time);
            CDLP_checker(graph, cdlp_result, cdlp_pass);

            result_all.push_back(std::make_pair("CDLP", std::to_string(gpu_cdlp_time)));
        }
        else
            result_all.push_back(std::make_pair("CDLP", "N/A"));

        graph.save_to_CSV(result_all, "../results/output.csv", "GPU");
    }
    freopen("CON", "r", stdin);

    return 0;
}
