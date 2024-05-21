#include <GPU_BFS.cuh>
#include <Union-Find.cuh>
#include <Workfront-Sweep.cuh>
#include <GPU_PageRank.cuh>
#include <GPU_Community_Detection.cuh>

#include <CPU_BFS.hpp>
#include <CPU_connected_components.hpp>
#include <CPU_shortest_paths.hpp>
#include <CPU_PageRank.hpp>
#include <CPU_Community_Detection.hpp>

#include <checker.hpp>

#include <time.h>
#include<math.h>
int main()
{
    std::string config_file;
    std::cout << "Enter the name of the configuration file:" << std::endl;
    // std::cin >> config_file;
    config_file="cit-Patents.properties";
    config_file = "../data/" + config_file;

    graph_structure<double> graph;
    graph.read_config(config_file);

    graph.load_LDBC();
    CSR_graph<double> csr_graph = graph.toCSR();
    std::cout << "Number of vertices: " << csr_graph.OUTs_Neighbor_start_pointers.size() << std::endl;
    std::cout << "Number of edges: " << csr_graph.OUTs_Edges.size() << std::endl;

    float elapsedTime = 0;
    float cpu_time=0;
    float gpu_time=0;
    clock_t start = clock(), end = clock();

    if (graph.sup_bfs) {
        std::vector<int> cpu_bfs_result;
        start = clock();
        cpu_bfs_result = CPU_BFS<double>(graph.OUTs, graph.bfs_src);
        end = clock();
        printf("CPU BFS cost time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

        std::vector<int> gpu_bfs_result;
        gpu_bfs_result = cuda_bfs(csr_graph, graph.bfs_src, &elapsedTime);
        elapsedTime = 0;
        cuda_bfs(csr_graph, graph.bfs_src, &elapsedTime);
        printf("GPU BFS cost time: %f ms\n", elapsedTime);
        elapsedTime = 0;

        bfs_checker(graph, cpu_bfs_result, gpu_bfs_result);
    }

    if (graph.sup_wcc) {
        std::vector<std::vector<int>> cpu_wcc_result;
        start = clock();
        cpu_wcc_result = CPU_connected_components<double>(graph.OUTs);
        end = clock();
        printf("CPU WCC cost time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

        std::vector<std::vector<int>> gpu_wcc_result;
        gpu_wcc_result = gpu_connected_components(csr_graph, &elapsedTime);
        elapsedTime = 0;
        gpu_connected_components(csr_graph, &elapsedTime);
        printf("GPU WCC cost time: %f ms\n", elapsedTime);
        elapsedTime = 0;

        wcc_checker(graph, cpu_wcc_result, gpu_wcc_result);
    }

    if (graph.sup_sssp) {
        start = clock();
        std::vector<double> cpu_sssp_result = CPU_shortest_paths(graph.OUTs, graph.sssp_src);
        end = clock();
        printf("CPU SSSP cost time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    
        elapsedTime = 0;
        std::vector<double> gpu_sssp_result(graph.V, 0);
        Workfront_Sweep(csr_graph, graph.sssp_src, gpu_sssp_result, &elapsedTime);
        printf("GPU SSSP cost time: %f ms\n", elapsedTime);

        sssp_checker(graph, cpu_sssp_result, gpu_sssp_result);
    }

    if (graph.sup_cdlp) {
        start = clock();
        vector<int> ans_cpu;
        CPU_Community_Detection(graph,ans_cpu);
        end = clock();
        cpu_time=(double)(end - start) / CLOCKS_PER_SEC * 1000;

        // start = clock();
        // CPU_Community_Detection(graph);
        // end = clock();
        // cpu_time=min(cpu_time,(float)(end - start) / CLOCKS_PER_SEC * 1000);

        // start = clock();
        // CPU_Community_Detection(graph);
        // end = clock();
        // cpu_time=min(cpu_time,(float)(end - start) / CLOCKS_PER_SEC * 1000);
        printf("CPU Community Detection cost time: %f ms\n", cpu_time);
        vector<int> ans_gpu;
        elapsedTime = 0;
        Community_Detection(graph, &elapsedTime,ans_gpu);
        printf("GPU Community Detection cost time: %f ms\n", elapsedTime);
        elapsedTime = 0;
        
        for(int i=0;i<3774768;i++){
            if(ans_cpu[i]!=ans_gpu[i])
                cout<<"diffrent at point : "<<i<<"  "<<ans_cpu[i]<<"   "<<ans_gpu[i]<<endl;
        }

    }

    return 0;
}
