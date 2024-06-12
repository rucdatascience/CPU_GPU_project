#include <GPU_BFS.cuh>
#include <GPU_connected_components.cuh>
#include <GPU_shortest_paths.cuh>
#include <GPU_PageRank.cuh>
#include <GPU_Community_Detection.cuh>

#include <CPU_BFS.hpp>
#include <CPU_connected_components.hpp>
#include <CPU_shortest_paths.hpp>
#include <CPU_PageRank.hpp>
#include <CPU_Community_Detection.hpp>
#include <cdlp_cpu.hpp>
#include <checker.hpp>
#include <checker_cpu.hpp>
#include <checker_gpu.hpp>

#include <time.h>

//one test one result file
void saveAsCSV(const unordered_map<string, string>& data, const string& filename);

//all test one result file
void writeToCSV(const unordered_map<string, string>& data, const string& filename);

int main()
{

    // auto begin = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();
    // double t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s

    std::string config_file = "datagen-7_5-fb.properties";//quick test
    // std::string config_file = "cit-Patents.properties";//quick test
    std::cout << "Enter the name of the configuration file:" << std::endl;
    // std::cin >> config_file;
    config_file = "../data/" + config_file;
    std::cout<<"config_file is:"<<config_file<<endl;

    graph_structure<double> graph;
    //Read the ldbc configuration file to obtain key parameter information in the file
    graph.read_config(config_file);
    //Read the vertex and edge files corresponding to the configuration file
    graph.load_LDBC();
    // The vertex information in graph is converted to csr format for storage
    CSR_graph<double> csr_graph = graph.toCSR();
    std::cout << "Number of vertices: " << csr_graph.OUTs_Neighbor_start_pointers.size() << std::endl;
    std::cout << "Number of edges: " << csr_graph.OUTs_Edges.size() << std::endl;

    float elapsedTime = 0;

    clock_t start = clock(), end = clock();

    unordered_map<string, string> umap_all_res;
    
    size_t lastSlashPos = config_file.find_last_of("/\\");
    size_t lastDotPos = config_file.find_last_of(".");
    string test_file_name = config_file.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);
    // cout<<"test_file_name:"<<test_file_name<<endl;

    umap_all_res.emplace("test_file_name", test_file_name);

    if (graph.sup_bfs) {
        int bfs_pass = 0;
        std::vector<int> cpu_bfs_result;
        start = clock();
        cpu_bfs_result = CPU_BFS<double>(graph.OUTs, graph.bfs_src);
        end = clock();
        double cpu_bfs_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("CPU BFS cost time: %f ms\n", cpu_bfs_time);

        std::vector<int> gpu_bfs_result;
        start = clock();
        gpu_bfs_result = cuda_bfs(csr_graph, graph.bfs_src, &elapsedTime);
        end = clock();
        double gpu_bfs_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;

        // // elapsedTime = 0;
        // cuda_bfs(csr_graph, graph.bfs_src, &elapsedTime);
        // double gpu_bfs_time;

        printf("GPU BFS cost time: %f ms\n", gpu_bfs_time);
        elapsedTime = 0;

        bfs_checker(graph, cpu_bfs_result, gpu_bfs_result, bfs_pass);
        string bfs_pass_label = "No";
        if(bfs_pass != 0) bfs_pass_label = "Yes";
        
        umap_all_res.emplace("cpu_bfs_time", std::to_string(cpu_bfs_time));
        umap_all_res.emplace("gpu_bfs_time", std::to_string(gpu_bfs_time));
        umap_all_res.emplace("bfs_pass_label", bfs_pass_label);
    }

    if (graph.sup_wcc) {
        int wcc_pass = 0;
        std::vector<std::vector<int>> cpu_wcc_result;
        start = clock();
        cpu_wcc_result = CPU_connected_components<double>(graph.OUTs);
        end = clock();
        double cpu_wcc_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("CPU WCC cost time: %f ms\n", cpu_wcc_time);

        std::vector<std::vector<int>> gpu_wcc_result;
        elapsedTime = 0;
        start = clock();
        gpu_wcc_result = gpu_connected_components(csr_graph, &elapsedTime);
        end = clock();
        double gpu_wcc_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("GPU WCC cost time: %f ms\n", gpu_wcc_time);

        wcc_checker(graph, cpu_wcc_result, gpu_wcc_result, wcc_pass);
        string wcc_pass_label = "No";
        if(wcc_pass != 0) wcc_pass_label = "Yes";

        umap_all_res.emplace("cpu_wcc_time", std::to_string(cpu_wcc_time));
        umap_all_res.emplace("gpu_wcc_time", std::to_string(gpu_wcc_time));
        umap_all_res.emplace("wcc_pass_label", wcc_pass_label);

    }

    if (graph.sup_sssp) {
        int sssp_pass = 0;
        start = clock();
        std::vector<double> cpu_sssp_result = CPU_shortest_paths(graph.OUTs, graph.sssp_src);
        end = clock();
        double cpu_sssp_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("CPU SSSP cost time: %f ms\n", cpu_sssp_time);
    
        elapsedTime = 0;
        std::vector<double> gpu_sssp_result(graph.V, 0);
        start = clock();
        gpu_shortest_paths(csr_graph, graph.sssp_src, gpu_sssp_result, &elapsedTime);
        end = clock();
        double gpu_sssp_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("GPU SSSP cost time: %f ms\n", gpu_sssp_time);

        sssp_checker(graph, cpu_sssp_result, gpu_sssp_result, sssp_pass);

        string sssp_pass_label = "No";
        if(sssp_pass != 0) sssp_pass_label = "Yes";

        umap_all_res.emplace("cpu_sssp_time", std::to_string(cpu_sssp_time));
        umap_all_res.emplace("gpu_sssp_time", std::to_string(gpu_sssp_time));
        umap_all_res.emplace("sssp_pass_label", sssp_pass_label);    
    }

    if (graph.sup_pr) {
        int pr_pass = 0;
        vector<double> cpu_pr_result, gpu_pr_result;
        start = clock();
        CPU_PageRank(graph, cpu_pr_result);
        end = clock();
        double cpu_pr_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        pr_checker_cpu(graph, cpu_pr_result, pr_pass);
        printf("CPU PageRank cost time: %f ms\n", cpu_pr_time);

        elapsedTime = 0;
        start = clock();
        gpu_PageRank(graph, &elapsedTime, gpu_pr_result);
        end = clock();
        double gpu_pr_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("GPU PageRank cost time: %f ms\n", gpu_pr_time);

        pr_checker(graph, cpu_pr_result, gpu_pr_result, pr_pass);

        string pr_pass_label = "No";
        if(pr_pass != 0) pr_pass_label = "Yes";

        umap_all_res.emplace("cpu_pr_time", std::to_string(cpu_pr_time));
        umap_all_res.emplace("gpu_pr_time", std::to_string(gpu_pr_time));
        umap_all_res.emplace("pr_pass_label", pr_pass_label);        
    }

    if (graph.sup_cdlp) {
        int cdlp_pass = 0;
        std::vector<int> ans_cpu, ans_gpu;
        start = clock();
        CPU_Community_Detection(graph, ans_cpu);
        end = clock();
        double cpu_cdlp_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        cdlp_check_cpu(graph, ans_cpu, cdlp_pass);
        printf("CPU Community Detection cost time: %f ms\n", cpu_cdlp_time);

        // start = clock();
        // vector<int> cdlp_result = community_detection_CDP(graph, graph.cdlp_max_its);
        // end = clock();
        // double time_cpu_cdlp = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        // printf("CPU_cdlp cost time: %f ms\n", time_cpu_cdlp);

        // cdlp_check_cpu(graph, cdlp_result, cdlp_pass);

        elapsedTime = 0;
        start = clock();
        gpu_Community_Detection(graph, &elapsedTime, ans_gpu);
        end = clock();
        double gpu_cdlp_time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        printf("GPU Community Detection cost time: %f ms\n", gpu_cdlp_time);
        cdlp_check_gpu(graph, ans_gpu, cdlp_pass);

        string cdlp_pass_label = "No";
        if(cdlp_pass != 0) cdlp_pass_label = "Yes";
        umap_all_res.emplace("cpu_cdlp_time", std::to_string(cpu_cdlp_time));
        umap_all_res.emplace("gpu_cdlp_time", std::to_string(gpu_cdlp_time));
        umap_all_res.emplace("cdlp_pass_label", cdlp_pass_label);        
    }

    
    time_t execute_time;
    time(&execute_time);
    umap_all_res.emplace("execute_time", std::to_string(execute_time));

    //store test file to .csv file
    string store_path = "../results/" + test_file_name + std::to_string(execute_time) + ".csv";
    // cout <<"result store path is:"<<store_path<<endl;
    saveAsCSV(umap_all_res, store_path);

    //save all test result to a test.csv file
    // string save_result = "../results/test.csv";
    // cout <<"result store path is:"<<save_result<<endl;
    // writeToCSV(umap_all_res, save_result);

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
        } else {
            file << endl;
        }
    }

    // write value
    for (auto it = data.begin(); it != data.end(); ++it) {
        file << it->second;
        if (next(it) != data.end()) {
            file << ",";
        } else {
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
            } else {  
                file << std::endl; 
            }  
        }  
    }  
  
    // write value（no matter what the pointer position）  
    for (auto it = data.begin(); it != data.end(); ++it) {  
        file << it->second;  
        if (std::next(it) != data.end()) { 
            file << "|";  
        } else {  
            file << std::endl;
        }  
    }

    file.close();
    cout << "Data writed to " << filename << " successfully." << endl;

}