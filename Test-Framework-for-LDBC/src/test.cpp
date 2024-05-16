//#include "../include/GPU_BFS.cuh"
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

#include <time.h>

int main()
{
    std::string config_file;
    std::cout << "Enter the name of the configuration file:" << std::endl;
    std::cin >> config_file;
    config_file = "../data/" + config_file;

    graph_structure<double> graph;
    graph.read_config(config_file);

    graph.load_LDBC();
    CSR_graph<double> csr_graph = graph.toCSR();
    std::cout << "Number of vertices: " << csr_graph.OUTs_Neighbor_start_pointers.size() << std::endl;
    std::cout << "Number of edges: " << csr_graph.OUTs_Edges.size() << std::endl;

    float elapsedTime = 0;

    clock_t start = clock(), end = clock();

    if (graph.sup_bfs) {
        start = clock();
        CPU_BFS<double>(graph.OUTs, graph.bfs_src);
        end = clock();
        printf("CPU BFS cost time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

        cuda_bfs(csr_graph, graph.bfs_src, &elapsedTime);
        elapsedTime = 0;
        cuda_bfs(csr_graph, graph.bfs_src, &elapsedTime);
        printf("GPU BFS cost time: %f ms\n", elapsedTime);
        elapsedTime = 0;
    }

    if (graph.sup_wcc) {
        start = clock();
        CPU_connected_components<double>(graph.OUTs);
        end = clock();
        printf("CPU WCC cost time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

        gpu_connected_components(csr_graph, &elapsedTime);
        elapsedTime = 0;
        gpu_connected_components(csr_graph, &elapsedTime);
        printf("GPU WCC cost time: %f ms\n", elapsedTime);
        elapsedTime = 0;
    }

    if (graph.sup_sssp) {
        start = clock();
        std::vector<double> sssp_result;
        CPU_shortest_paths(graph.OUTs, graph.sssp_src, sssp_result);
        end = clock();
        printf("CPU SSSP cost time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    
        elapsedTime = 0;
        std::vector<double> gpu_sssp_result(graph.V, 0);
        Workfront_Sweep(csr_graph, graph.sssp_src, gpu_sssp_result, &elapsedTime);
        printf("GPU SSSP cost time: %f ms\n", elapsedTime);
    }

    if (graph.sup_pr) {
        start = clock();
        CPU_PageRank(graph);
        end = clock();
        printf("CPU PageRank cost time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

        elapsedTime = 0;
        PageRank(graph, &elapsedTime);
        printf("GPU PageRank cost time: %f ms\n", elapsedTime);
    }

    /*if (graph.sup_cdlp) {
        start = clock();
        CPU_Community_Detection(graph);
        end = clock();
        printf("CPU Community Detection cost time: %f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

        elapsedTime = 0;
        Community_Detection(graph, &elapsedTime);
        printf("GPU Community Detection cost time: %f ms\n", elapsedTime);
        elapsedTime = 0;
    }*/

    return 0;
}