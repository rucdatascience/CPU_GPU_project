#include <GPU_BFS.cuh>
#include <GPU_connected_components.cuh>
#include <GPU_shortest_paths.cuh>
#include <GPU_PageRank.cuh>
#include "GPU_Community_Detection.cuh"
#include <chrono>
// #include <checker.hpp>
#include <checkerLDBC.hpp>
#include <time.h>
//one test one result file
#include "../include/UserInfo.hpp"
void saveAsCSV(const unordered_map<string, string>& data, const string& filename);

//all test one result file
void writeToCSV(const unordered_map<string, string>& data, const string& filename);


int main()
{

    // std::string config_file = "datagen-7_5-fb.properties";//quick test
    // std::string config_file = "cit-Patents.properties";//quick test

    vector<string> datas = {"cit-Patents.properties" , "datagen-7_5-fb.properties" };
    
    for (string config_file : datas) {

        config_file = "../data/" + config_file;
        std::cout << "config_file is:" << config_file << endl;

        // graph_structure<double> graph;
        LDBC<double> graph;
        graph.read_config(config_file); //Read the ldbc configuration file to obtain key parameter information in the file

        auto begin = std::chrono::high_resolution_clock::now();
        graph.load_LDBC(); //Read the vertex and edge files corresponding to the configuration file, // The vertex information in graph is converted to csr format for storage   
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

        // vector<string> userNameVec;
        // std::vector<UserInfo> users = readUserFile();
        // for(auto & it : graph.vertex_id_to_str){
        //     userNameVec.push_back(getUserNameById(it));//Note that there is no user name in the current file
        // }

        float elapsedTime = 0;
        unordered_map<string, string> umap_all_res;
        size_t lastSlashPos = config_file.find_last_of("/\\");
        size_t lastDotPos = config_file.find_last_of(".");
        string test_file_name = config_file.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);
        umap_all_res.emplace("test_file_name", test_file_name);
        if (graph.sup_cdlp) {

            if (0) {

                int cdlp_pass = 0;
                std::vector<string> ans_gpu(graph.size());
                begin = std::chrono::high_resolution_clock::now();
                CDLP_GPU(graph, csr_graph,ans_gpu);
                end = std::chrono::high_resolution_clock::now();
                double gpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU Community Detection cost time: %f s\n", gpu_cdlp_time);
                /*check*/
                // cdlp_check(graph, ans_cpu, cdlp_pass);
                // std::unordered_map<string, string> gpuCDLP4name = getGPUCDLP(userNameVec, graph, csr_graph);
                cdlp_ldbc_check(graph, ans_gpu, cdlp_pass);
            }
        }
        if (graph.sup_pr) {
            int pr_pass = 0;

            if (1) {
                elapsedTime = 0;
                vector<double> gpu_pr_result;
                begin = std::chrono::high_resolution_clock::now();
                GPU_PR(graph, &elapsedTime, gpu_pr_result,csr_graph.in_pointer,csr_graph.out_pointer,csr_graph.in_edge,csr_graph.out_edge);
                end = std::chrono::high_resolution_clock::now();
                double gpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU PageRank cost time: %f s\n", gpu_pr_time);

                /*check*/
                // pr_checker(graph, gpu_pr_result, pr_pass);
                // std::unordered_map<string, double> gpuPR4name = getGPUPR(userNameVec, graph, csr_graph);
                pr_ldbc_checker(graph, gpu_pr_result, pr_pass);
            }
        }

        if (graph.sup_bfs) {
            int bfs_pass = 0;
        
            if (1) {
                std::vector<int> gpu_bfs_result;
                begin = std::chrono::high_resolution_clock::now();
                gpu_bfs_result = cuda_bfs(csr_graph, graph.bfs_src, &elapsedTime);
                end = std::chrono::high_resolution_clock::now();
                
                double gpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU BFS cost time: %f s\n", gpu_bfs_time);

                /*check*/
                // bfs_checker(graph, gpu_bfs_result, bfs_pass);
                // std::unordered_map<string, int> gpuBFS4name = getGPUBFS(userNameVec, graph, csr_graph);
                bfs_ldbc_checker(graph, gpu_bfs_result, bfs_pass);
            }
        }

        if (graph.sup_wcc) {
            int wcc_pass = 0;
            std::vector<std::vector<int>> cpu_wcc_result;

            if (1) {
                std::vector<std::vector<int>> gpu_wcc_result;
                elapsedTime = 0;
                auto begin = std::chrono::high_resolution_clock::now();
                gpu_wcc_result = gpu_connected_components(csr_graph, &elapsedTime);
                auto end = std::chrono::high_resolution_clock::now();
                double gpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU WCC cost time: %f s\n", gpu_wcc_time);

                /*check*/
                // wcc_checker(graph, gpu_wcc_result, wcc_pass);
                wcc_ldbc_checker(graph, gpu_wcc_result, wcc_pass);
            }
        }

        if (graph.sup_sssp) {
            int sssp_pass = 0;

            if (1) {
                std::vector<double> gpu_sssp_result(graph.V, 0);
                begin = std::chrono::high_resolution_clock::now();
                gpu_shortest_paths(csr_graph, graph.sssp_src, gpu_sssp_result, &elapsedTime);
                end = std::chrono::high_resolution_clock::now();
                double gpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
                printf("GPU SSSP cost time: %f s\n", gpu_sssp_time);

                /*check*/
                // sssp_checker(graph, gpu_sssp_result, sssp_pass);
                // std::unordered_map<string, double> gpuSSSP4name = getGPUSSSP(userNameVec, graph, csr_graph);
                sssp_ldbc_checker(graph, gpu_sssp_result, sssp_pass);
            }
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

// Save the unordered_map as a csv file
void saveAsCSV(const unordered_map<string, string>& data, const string& filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cout << "Failed to open file for writing: " << filename << endl;
        return;
    }

    // write title
    for (auto it = data.begin(); it != data.end(); ++it) {
        file << it->first;
        if (next(it) != data.end()) {
            file << ",";
        }
        else {
            file << endl;
        }
    }

    // write value
    for (auto it = data.begin(); it != data.end(); ++it) {
        file << it->second;
        if (next(it) != data.end()) {
            file << ",";
        }
        else {
            file << endl;
        }
    }

    file.close();
    cout << "Data saved to " << filename << " successfully." << endl;
}

// write all result to a *.csv file
void writeToCSV(const unordered_map<string, string>& data, const string& filename) {
    // Creates a file or opens an existing file and allows you to append content
    ofstream file(filename, ios::out | ios::app);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    // Check if the file is empty, and insert a header line if it is
    file.seekp(0, ios::end); // Locate end of file

    // write title（if file point in the start position）  
    if (file.tellp() == 0) {
        // write title  
        for (auto it = data.begin(); it != data.end(); ++it) {
            file << it->first;
            if (std::next(it) != data.end()) {
                file << "|";
            }
            else {
                file << std::endl;
            }
        }
    }

    // write value（no matter what the pointer position）  
    for (auto it = data.begin(); it != data.end(); ++it) {
        file << it->second;
        if (std::next(it) != data.end()) {
            file << "|";
        }
        else {
            file << std::endl;
        }
    }

    file.close();
    cout << "Data writed to " << filename << " successfully." << endl;

}