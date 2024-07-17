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

    // std::string config_file = "datagen-7_5-fb.properties";//quick test
    // std::string config_file = "cit-Patents.properties";//quick test

    vector<string> datas = {"cit-Patents.properties" , "datagen-7_5-fb.properties" };
    
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
            /*if (1) {
                std::vector<int> gpu_bfs_result;
                begin = std::chrono::high_resolution_clock::now();
                gpu_bfs_result = cuda_bfs(csr_graph, graph.bfs_src);
                end = std::chrono::high_resolution_clock::now();
                double gpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU BFS cost time: %f s\n", gpu_bfs_time);
                // bfs_checker(graph, gpu_bfs_result, bfs_pass);
                bfs_ldbc_checker(graph, gpu_bfs_result, bfs_pass);
            }

            if(1){
                std::vector<std::string> gpu_bfs_result_v2;
                begin = std::chrono::high_resolution_clock::now();
                gpu_bfs_result_v2 = cuda_bfs_v2(graph, csr_graph);
                end = std::chrono::high_resolution_clock::now();
                double gpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU BFS V2 cost time: %f s\n", gpu_bfs_time);
                bfs_ldbc_checker_v2(graph, gpu_bfs_result_v2, bfs_pass);
            }*/
        }
        //continue;

        if (graph.sup_sssp) {
            bool sssp_pass = 0;

            std::vector<std::pair<std::string, double>> sssp_result;
            begin = std::chrono::high_resolution_clock::now();
            sssp_result = Cuda_SSSP(graph, csr_graph, graph.sssp_src_name, 10000000000);
            end = std::chrono::high_resolution_clock::now();
            double gpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("GPU SSSP cost time: %f s\n", gpu_sssp_time);
            SSSP_checker(graph, sssp_result, sssp_pass);

            /*if (1) {
                std::vector<double> gpu_sssp_result(graph.V, 0);
                begin = std::chrono::high_resolution_clock::now();
                gpu_shortest_paths(csr_graph, graph.sssp_src, gpu_sssp_result, 0, 10000000000);
                end = std::chrono::high_resolution_clock::now();
                double gpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU SSSP cost time: %f s\n", gpu_sssp_time);
                // sssp_checker(graph, gpu_sssp_result, sssp_pass);
                sssp_ldbc_checker(graph, gpu_sssp_result, sssp_pass);
                graph.res_trans_id_val(gpu_sssp_result);
            }*/

            /*if(1){
                begin = std::chrono::high_resolution_clock::now();
                std::vector<std::string> gpu_sssp_result_v2 = gpu_shortest_paths_v2(graph, csr_graph);
                end = std::chrono::high_resolution_clock::now();
                double gpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU SSSP v2 cost time: %f s\n", gpu_sssp_time);
                sssp_ldbc_checker_v2(graph, gpu_sssp_result_v2, sssp_pass);
            }*/
        }

        if (graph.sup_wcc) {
            bool wcc_pass = false;

            std::vector<std::pair<std::string, std::string>> wcc_result;
            begin = std::chrono::high_resolution_clock::now();
            wcc_result = Cuda_WCC(graph, csr_graph);
            end = std::chrono::high_resolution_clock::now();
            double gpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("GPU WCC cost time: %f s\n", gpu_wcc_time);
            WCC_checker(graph, wcc_result, wcc_pass);

            /*std::vector<std::vector<int>> gpu_wcc_result;
            if (1) {
                elapsedTime = 0;
                begin = std::chrono::high_resolution_clock::now();
                gpu_wcc_result = gpu_connected_components(csr_graph);
                end = std::chrono::high_resolution_clock::now();
                double gpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU WCC cost time: %f s\n", gpu_wcc_time);
                // wcc_checker(graph, gpu_wcc_result, wcc_pass);
                wcc_ldbc_checker(graph, gpu_wcc_result, wcc_pass);
                graph.res_trans_id_id(gpu_wcc_result);
            }*/
            
            /*if(1){
                elapsedTime = 0;
                begin = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<std::string>> gpu_wcc_result_v2 = gpu_connected_components_v2(csr_graph, &elapsedTime);
                end = std::chrono::high_resolution_clock::now();
                double gpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU WCC V2 cost time: %f s\n", gpu_wcc_time);
                wcc_ldbc_checker_v2(graph, gpu_wcc_result_v2, wcc_pass);
            }*/
        }

        if (graph.sup_pr) {
            bool pr_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, double>> pr_result = Cuda_PR(graph, csr_graph, graph.pr_its, graph.pr_damping);
            end = std::chrono::high_resolution_clock::now();
            double gpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("GPU PageRank cost time: %f s\n", gpu_pr_time);
            PR_checker(graph, pr_result, pr_pass);
            /*int pr_pass = 0;
            if (1) {
                elapsedTime = 0;
                vector<double> gpu_pr_result;
                begin = std::chrono::high_resolution_clock::now();
                GPU_PR(graph, &elapsedTime, gpu_pr_result,csr_graph.in_pointer,csr_graph.out_pointer,csr_graph.in_edge,csr_graph.out_edge);
                end = std::chrono::high_resolution_clock::now();
                double gpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU PageRank cost time: %f s\n", gpu_pr_time);
                // pr_checker(graph, gpu_pr_result, pr_pass);
                pr_ldbc_checker(graph, gpu_pr_result, pr_pass);
            }

            if(1){
                vector<std::string> gpu_pr_result_v2;
                begin = std::chrono::high_resolution_clock::now();

                GPU_PR_v3(graph, &elapsedTime,gpu_pr_result_v2,csr_graph.in_pointer,csr_graph.out_pointer,csr_graph.in_edge,csr_graph.out_edge);
                
                end = std::chrono::high_resolution_clock::now();
                double gpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU PageRank V2 cost time: %f s\n", gpu_pr_time);
                pr_ldbc_checker_v2(graph, gpu_pr_result_v2, pr_pass);
            }*/
        }

        if (graph.sup_cdlp) {
            bool cdlp_pass = false;

            begin = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<std::string, std::string>> cdlp_result = Cuda_CDLP(graph, csr_graph, graph.cdlp_max_its);
            end = std::chrono::high_resolution_clock::now();
            double gpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            printf("GPU Community Detection cost time: %f s\n", gpu_cdlp_time);
            CDLP_checker(graph, cdlp_result, cdlp_pass);

            /*if (1) {

                std::vector<string> ans_gpu(graph.size());
                begin = std::chrono::high_resolution_clock::now();
                CDLP_GPU(graph, csr_graph,ans_gpu);
                end = std::chrono::high_resolution_clock::now();
                double gpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU Community Detection cost time: %f s\n", gpu_cdlp_time);
                // cdlp_check(graph, ans_cpu, cdlp_pass);
                cdlp_ldbc_check(graph, ans_gpu, cdlp_pass);

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
