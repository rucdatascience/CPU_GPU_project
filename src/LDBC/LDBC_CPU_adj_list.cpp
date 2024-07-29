#include <chrono>
#include <CPU_adj_list/algorithm/CPU_BFS.hpp>
#include <CPU_adj_list/algorithm/CPU_connected_components.hpp>
#include <CPU_adj_list/algorithm/CPU_shortest_paths.hpp>
#include <CPU_adj_list/algorithm/CPU_PageRank.hpp>
#include <CPU_adj_list/algorithm/CPU_Community_Detection.hpp>
#include <LDBC/checker.hpp>
#include <LDBC/ldbc.hpp>
#include <time.h>

int main()
{
    ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);
    
    vector<string> datas = {"datagen-7_5-fb.properties"};

    freopen("../input.txt", "r", stdin);

    for (string config_file : datas) {

        std::string config_file_path;
        std::cout << "Please input the config file path: ";
        std::cin >> config_file_path;
        
        LDBC<double> graph;
        graph.read_config(config_file_path); //Read the ldbc configuration file to obtain key parameter information in the file

        auto begin = std::chrono::high_resolution_clock::now();
        graph.load_graph(); //Read the vertex and edge files corresponding to the configuration file, // The vertex information in graph is converted to csr format for storage   
        auto end = std::chrono::high_resolution_clock::now();
        double load_ldbc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
        printf("load_ldbc_time cost time: %f s\n", load_ldbc_time);

        float elapsedTime = 0;
        unordered_map<string, string> umap_all_res;
        size_t lastSlashPos = config_file.find_last_of("/\\");
        size_t lastDotPos = config_file.find_last_of(".");
        string test_file_name = config_file.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);
        umap_all_res.emplace("test_file_name", test_file_name);

        std::vector<std::pair<std::string, std::string>> result_all;

        if (graph.sup_bfs) {
            bool bfs_pass = false;

            std::vector<std::pair<std::string, int>> cpu_bfs_result;
            begin = std::chrono::high_resolution_clock::now();
            cpu_bfs_result = CPU_Bfs(graph, graph.bfs_src_name);
            end = std::chrono::high_resolution_clock::now();
            double cpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU BFS cost time: %f s\n", cpu_bfs_time);
            Bfs_checker(graph, cpu_bfs_result, bfs_pass);

            result_all.push_back(std::make_pair("BFS", std::to_string(cpu_bfs_time)));
        }
        else
            result_all.push_back(std::make_pair("BFS", "N/A"));

        if (graph.sup_sssp) {
            bool sssp_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, double>> cpu_sssp_result = CPU_SSSP(graph, graph.sssp_src_name);
            end = std::chrono::high_resolution_clock::now();
            double cpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU SSSP cost time: %f s\n", cpu_sssp_time);
            SSSP_checker(graph, cpu_sssp_result, sssp_pass);

            result_all.push_back(std::make_pair("SSSP", std::to_string(cpu_sssp_time)));
        }
        else
            result_all.push_back(std::make_pair("SSSP", "N/A"));

        if (graph.sup_wcc) {
            bool wcc_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, std::string>> cpu_wcc_result = CPU_WCC(graph);
            end = std::chrono::high_resolution_clock::now();
            double cpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU WCC cost time: %f s\n", cpu_wcc_time);
            WCC_checker(graph, cpu_wcc_result, wcc_pass);

            result_all.push_back(std::make_pair("WCC", std::to_string(cpu_wcc_time)));
        }
        else
            result_all.push_back(std::make_pair("WCC", "N/A"));

        if (graph.sup_pr) {
            bool pr_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, double>> cpu_pr_result = CPU_PR(graph, graph.pr_its, graph.pr_damping);
            end = std::chrono::high_resolution_clock::now();
            double cpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU PageRank cost time: %f s\n", cpu_pr_time);
            PR_checker(graph, cpu_pr_result, pr_pass);

            result_all.push_back(std::make_pair("PageRank", std::to_string(cpu_pr_time)));
        }
        else
            result_all.push_back(std::make_pair("PageRank", "N/A"));

        if (graph.sup_cdlp) {
            bool cdlp_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, std::string>> cpu_cdlp_result = CPU_CDLP(graph, graph.cdlp_max_its);
            end = std::chrono::high_resolution_clock::now();
            double cpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU Community Detection cost time: %f s\n", cpu_cdlp_time);
            CDLP_checker(graph, cpu_cdlp_result, cdlp_pass);

            result_all.push_back(std::make_pair("CommunityDetection", std::to_string(cpu_cdlp_time)));
        }
        else
            result_all.push_back(std::make_pair("CommunityDetection", "N/A"));


        graph.save_to_CSV(result_all, "../results/output.csv", "CPU");
    }
    freopen("CON", "r", stdin);

    return 0;
}
