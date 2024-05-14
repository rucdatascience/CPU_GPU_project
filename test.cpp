//#include "../include/GPU_BFS.cuh"
#include <GPU_BFS.cuh>
#include <Union-Find.cuh>
#include <Workfront-Sweep.cuh>

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
    if (graph.sup_bfs) {
        cuda_bfs(csr_graph, graph.bfs_src, &elapsedTime);
        elapsedTime = 0;
        cuda_bfs(csr_graph, graph.bfs_src, &elapsedTime);
        printf("GPU BFS cost time: %f ms\n", elapsedTime);
        elapsedTime = 0;
    }

    if (graph.sup_wcc) {
        gpu_connected_components(csr_graph, &elapsedTime);
        elapsedTime = 0;
        gpu_connected_components(csr_graph, &elapsedTime);
        printf("GPU WCC cost time: %f ms\n", elapsedTime);
        elapsedTime = 0;
    }

    if (graph.sup_sssp) {
        std::vector<double> sssp_result(graph.V, 0);
        Workfront_Sweep(csr_graph, graph.sssp_src, sssp_result, &elapsedTime);
        printf("GPU SSSP cost time: %f ms\n", elapsedTime);
        elapsedTime = 0;
    }

    return 0;
}