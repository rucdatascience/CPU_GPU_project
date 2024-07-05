#include <chrono>
#include "CPU_BFS.hpp"
#include "CPU_connected_components.hpp"
#include "CPU_shortest_paths.hpp"
#include "CPU_PageRank.hpp"
#include "CPU_Community_Detection_update.hpp"
#include "CPU_Community_Detection.hpp"
// #include "../include/checker.hpp"
#include "../include/ldbc.hpp"
#include "../include/checkerLDBC.hpp"
#include <time.h>
#include <map>
#include "../include/resultVSldbc.hpp"

//one test one result file
void saveAsCSV(const unordered_map<string, string>& data, const string& filename);

//all test one result file
void writeToCSV(const unordered_map<string, string>& data, const string& filename);


int main()
{

    // std::string config_file = "datagen-7_5-fb.properties";//quick test
    // std::string config_file = "cit-Patents.properties";//quick test

    vector<string> datas = {"cit-Patents.properties", "datagen-7_5-fb.properties"  };

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


        float elapsedTime = 0;
        unordered_map<string, string> umap_all_res;
        size_t lastSlashPos = config_file.find_last_of("/\\");
        size_t lastDotPos = config_file.find_last_of(".");
        string test_file_name = config_file.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);
        umap_all_res.emplace("test_file_name", test_file_name);

        if (graph.sup_bfs) {
            int bfs_pass = 0;

            if (1) {
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
            }
        }

        if (graph.sup_sssp) {
            int sssp_pass = 0;

            if (1) {
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
            }

          
        }

        if (graph.sup_wcc) {
            int wcc_pass = 0;

            if (1) {
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
            }
           
        }

        if (graph.sup_pr) {
            int pr_pass = 0;

            if (1) {
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
            }

           
        }

        if (graph.sup_cdlp) {

            if (1) {
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
