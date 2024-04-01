#include "GPU_BFS.cuh"

__global__ void bfs_kernel(int* edges, int* start, int* visited, int* queue, int* next_queue, int* queue_size, int* next_queue_size, int max_depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < *queue_size) {
        int vertex = queue[tid];
        int depth = visited[vertex];
        for (int edge = start[vertex]; edge < start[vertex + 1]; edge++) {
            int neighbor = edges[edge];
            if (visited[neighbor] >= max_depth && depth < max_depth) {
                visited[neighbor] = depth + 1;
                int pos = atomicAdd(next_queue_size, 1);
                next_queue[pos] = neighbor;
            }
        }
    }
}

float elapsedTime = 0.0;

template <typename T>
std::vector<int> cuda_bfs(ARRAY_graph<T>& input_graph, int source_vertex, int max_depth) {
    int V = input_graph.Neighbor_start_pointers.size() - 1;
    int E = input_graph.Edges.size();

    std::vector<int> depth(V, max_depth);

    if (source_vertex < 0 || source_vertex >= V) {
        fprintf(stderr, "Invalid source vertex\n");
        return depth;
    }

    int* visited;
    int* queue, * next_queue;
    int* queue_size, * next_queue_size;

    int* edges, * start;

    cudaMallocManaged((void**)&visited, V * sizeof(int));
    cudaMallocManaged((void**)&queue, V * sizeof(int));
    cudaMallocManaged((void**)&next_queue, V * sizeof(int));
    cudaMallocManaged((void**)&queue_size, sizeof(int));
    cudaMallocManaged((void**)&next_queue_size, sizeof(int));
    cudaMallocManaged((void**)&edges, E * sizeof(int));
    cudaMallocManaged((void**)&start, V * sizeof(int));

    cudaMemcpy(edges, input_graph.Edges.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(start, input_graph.Neighbor_start_pointers.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    queue[0] = source_vertex;
    for (int i = 0; i < V; i++)
        visited[i] = max_depth;
    visited[source_vertex] = 0;
    *queue_size = 1, *next_queue_size = 1;

    int threadsPerBlock = 1024;
    int numBlocks = 0;

    cudaEvent_t start_clock, stop_clock;

    cudaEventCreate(&start_clock);
    cudaEventCreate(&stop_clock);
    cudaEventRecord(start_clock, 0);

    while (*queue_size > 0) {
        numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;

        bfs_kernel << <numBlocks, threadsPerBlock >> > (edges, start, visited, queue, next_queue, queue_size, next_queue_size, max_depth);
        cudaDeviceSynchronize();

        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return depth;
        }

        std::swap(queue, next_queue);
        *queue_size = *next_queue_size;
        *next_queue_size = 0;
    }

    cudaEventRecord(stop_clock, 0);
    cudaEventSynchronize(stop_clock);
    cudaEventElapsedTime(&elapsedTime, start_clock, stop_clock);

    cudaEventDestroy(start_clock);
    cudaEventDestroy(stop_clock);

    cudaMemcpy(depth.data(), visited, V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(visited);
    cudaFree(queue);
    cudaFree(next_queue);
    cudaFree(queue_size);
    cudaFree(next_queue_size);
    cudaFree(edges);
    cudaFree(start);

    return depth;
}

int main()
{
    std::string file_path;
    std::cout << "Please input the file path of the graph: ";
    std::cin >> file_path;
    graph_v_of_v<int> graph;
    graph.txt_read(file_path);
    ARRAY_graph<int> array_graph = graph.toARRAY();
    int V = array_graph.Neighbor_start_pointers.size();
    cuda_bfs(array_graph, 0);
    float sum = 0;
    for (int i = 0; i < V; i++) {
        cuda_bfs(array_graph, i);
        sum += elapsedTime;
        elapsedTime = 0;
    }
    printf("GPU average cost time: %f ms\n", sum / V);
    return 0;
}

/*

nvcc -O3 -std=c++17 -o GPU_BFS.out GPU_BFS.cu
./GPU_BFS.out
rm GPU_BFS.out

*/