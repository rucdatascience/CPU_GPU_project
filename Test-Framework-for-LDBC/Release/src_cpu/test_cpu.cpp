#include <chrono>
#include "CPU_BFS.hpp"
#include "CPU_connected_components.hpp"
#include "CPU_shortest_paths.hpp"
#include "CPU_PageRank.hpp"
#include "CPU_Community_Detection.hpp"
#include <checker.hpp>
#include <time.h>

int main()
{

    // std::string config_file = "datagen-7_5-fb.properties";//quick test
    // std::string config_file = "cit-Patents.properties";//quick test

    vector<string> datas = {"cit-Patents.properties", "datagen-7_5-fb.properties"  };

    for (string config_file : datas) {

        config_file = "../data/" + config_file;
        std::cout << "config_file is:" << config_file << endl;

        // graph_structure<double> graph;
        
        graph_structure<double> graph;
        graph.read_config(config_file); //Read the ldbc configuration file to obtain key parameter information in the file

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

        if (graph.sup_bfs) {
            bool bfs_pass = false;

            std::vector<std::pair<std::string, int>> cpu_bfs_result;
            begin = std::chrono::high_resolution_clock::now();
            cpu_bfs_result = CPU_Bfs(graph, graph.bfs_src_name);
            end = std::chrono::high_resolution_clock::now();
            double cpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU BFS cost time: %f s\n", cpu_bfs_time);
            Bfs_checker(graph, cpu_bfs_result, bfs_pass);
            /*if (1) {
                std::vector<int> cpu_bfs_result;
                auto begin = std::chrono::high_resolution_clock::now();
                cpu_bfs_result = CPU_BFS<double>(graph.OUTs, graph.bfs_src);
                auto end = std::chrono::high_resolution_clock::now();
                double cpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("CPU BFS cost time: %f s\n", cpu_bfs_time);
                // bfs_checker(graph, cpu_bfs_result, bfs_pass);
                bfs_ldbc_checker(graph, cpu_bfs_result, bfs_pass);
            }

            if(1){
                std::vector<std::string> cpu_bfs_result_v2;
                begin = std::chrono::high_resolution_clock::now();
                cpu_bfs_result_v2 = CPU_BFS_v2(graph);
                end = std::chrono::high_resolution_clock::now();
                double cpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("CPU BFS V2 cost time: %f s\n", cpu_bfs_time);
                bfs_ldbc_checker_v2(graph, cpu_bfs_result_v2, bfs_pass);
            }*/
        }

        if (graph.sup_sssp) {
            bool sssp_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, double>> cpu_sssp_result = CPU_SSSP(graph, graph.sssp_src_name);
            end = std::chrono::high_resolution_clock::now();
            double cpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU SSSP cost time: %f s\n", cpu_sssp_time);
            SSSP_checker(graph, cpu_sssp_result, sssp_pass);

            /*if (1) {
                auto begin = std::chrono::high_resolution_clock::now();
                std::vector<double> cpu_sssp_result = CPU_shortest_paths(graph.OUTs, graph.sssp_src);
                auto end = std::chrono::high_resolution_clock::now();
                double cpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("CPU SSSP cost time: %f s\n", cpu_sssp_time);
                // sssp_checker(graph, cpu_sssp_result, sssp_pass);
                sssp_ldbc_checker(graph, cpu_sssp_result, sssp_pass);

            }

            if(1){
                begin = std::chrono::high_resolution_clock::now();
                std::vector<std::string> cpu_sssp_result_v2 = CPU_shortest_paths_v2(graph);
                end = std::chrono::high_resolution_clock::now();
                double cpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("CPU SSSP V2 cost time: %f s\n", cpu_sssp_time);
                sssp_ldbc_checker_v2(graph, cpu_sssp_result_v2, sssp_pass);
            }*/

          
        }

        if (graph.sup_wcc) {
            bool wcc_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, std::string>> cpu_wcc_result = CPU_WCC(graph);
            end = std::chrono::high_resolution_clock::now();
            double cpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU WCC cost time: %f s\n", cpu_wcc_time);
            WCC_checker(graph, cpu_wcc_result, wcc_pass);

            /*if (1) {
                std::vector<std::vector<int>> cpu_wcc_result;
                begin = std::chrono::high_resolution_clock::now();
                cpu_wcc_result = CPU_connected_components<double>(graph.OUTs, graph.INs);
                end = std::chrono::high_resolution_clock::now();
                double cpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("CPU WCC cost time: %f s\n", cpu_wcc_time);
                // wcc_checker(graph, cpu_wcc_result, wcc_pass);
                wcc_ldbc_checker(graph, cpu_wcc_result, wcc_pass);
            }

            if(1){
                std::vector<std::vector<std::string>> cpu_wcc_result_v2;
                begin = std::chrono::high_resolution_clock::now();
                cpu_wcc_result_v2 = CPU_connected_components_v2(graph);

                end = std::chrono::high_resolution_clock::now();
                double cpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("CPU WCC V2 cost time: %f s\n", cpu_wcc_time);
                wcc_ldbc_checker_v2(graph, cpu_wcc_result_v2, wcc_pass);
            }*/
           
        }

        if (graph.sup_pr) {
            bool pr_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, double>> cpu_pr_result = CPU_PR(graph, graph.pr_its, graph.pr_damping);
            end = std::chrono::high_resolution_clock::now();
            double cpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU PageRank cost time: %f s\n", cpu_pr_time);
            PR_checker(graph, cpu_pr_result, pr_pass);

            /*if (1) {
                vector<double> cpu_pr_result;
                begin = std::chrono::high_resolution_clock::now();
                cpu_pr_result = PageRank(graph.INs, graph.OUTs, graph.pr_damping, graph.pr_its);
                end = std::chrono::high_resolution_clock::now();
                double cpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s      
                printf("CPU PageRank cost time: %f s\n", cpu_pr_time);
                // pr_checker(graph, cpu_pr_result, pr_pass);
                pr_ldbc_checker(graph, cpu_pr_result, pr_pass);
                
                
            }

            if(1){
                begin = std::chrono::high_resolution_clock::now();
                vector<std::string> cpu_pr_result_v2 = PageRank_v2(graph);
                end = std::chrono::high_resolution_clock::now();
                double cpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s      
                printf("CPU PageRank V2 cost time: %f s\n", cpu_pr_time);
                pr_ldbc_checker_v2(graph, cpu_pr_result_v2, pr_pass);
            }*/

           
        }

        if (graph.sup_cdlp) {
            bool cdlp_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, std::string>> cpu_cdlp_result = CPU_CDLP(graph, graph.cdlp_max_its);
            end = std::chrono::high_resolution_clock::now();
            double cpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("CPU Community Detection cost time: %f s\n", cpu_cdlp_time);
            CDLP_checker(graph, cpu_cdlp_result, cdlp_pass);

            /*if (1) {
                int cdlp_pass = 0;
                std::vector<string> ans_cpu;

                int N = graph.vertex_id_to_str.size();
                vector<long long int> vertex_IDs_forCD(N);
                for (int i = N - 1; i >= 0; i--) {
                    vertex_IDs_forCD[i] = strtoll(graph.vertex_id_to_str[i].c_str(), NULL, 10);
                }

                begin = std::chrono::high_resolution_clock::now();
                ans_cpu = CDLP(graph, graph.cdlp_max_its);
                end = std::chrono::high_resolution_clock::now();
                double cpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("CPU Community Detection cost time: %f s\n", cpu_cdlp_time);
                // cdlp_check(graph, ans_cpu, cdlp_pass);
                cdlp_ldbc_check(graph, ans_cpu, cdlp_pass);
                
            }*/

        }


        time_t execute_time;
        time(&execute_time);
        umap_all_res.emplace("execute_time", std::to_string(execute_time));

        //store test file to .csv file
        // string store_path = "../results/" + test_file_name + std::to_string(execute_time) + ".csv";
        // cout <<"result store path is:"<<store_path<<endl;
        // saveAsCSV(umap_all_res, store_path);
    }

    return 0;
}
