#include <GPU_BFS.cuh>

__global__ void bfs_Relax(int* start, int* edge, int* depth, int* visited, int* queue, int* queue_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *queue_size) {
        int v = queue[idx];

        for (int i = start[v]; i < start[v + 1]; i++) {
            int new_v = edge[i];

            int new_depth = depth[v] + 1;

            int old = atomicMin(&depth[new_v], new_depth);

            if (old <= new_depth)
				continue;

            atomicExch(&visited[new_v], 1);
        }
    }
}

__global__ void bfs_CompactQueue(int V, int* next_queue, int* next_queue_size, int* visited) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V && visited[idx]) {
        int pos = atomicAdd(next_queue_size, 1);
        next_queue[pos] = idx;
        visited[idx] = 0;
    }
}

//It's not that the CPU tasks are assigned to the GPU, but rather that the GPU determines which part of the task to complete based on its own ID number
__global__ void bfs_kernel(int* edges, int* start, int* visited, int* queue, int* next_queue, int* queue_size, int* next_queue_size, int max_depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
   // Grid is divided into 1 dimension, Block is divided into 1 dimension.
    if (tid < *queue_size) {
        int vertex = queue[tid];
        int depth = visited[vertex];
        for (int edge = start[vertex]; edge < start[vertex + 1]; edge++) {
            //The traversal range of node edges is given by the start array
//Traverse adjacent edges
            int neighbor = edges[edge];
            if (visited[neighbor] >= max_depth && depth < max_depth) {
                visited[neighbor] = depth + 1;
                int pos = atomicAdd(next_queue_size, 1);//AtomicAdd is an atomic addition function in CUDA, used to ensure data consistency and correctness when multiple threads modify the same global variable simultaneously.
                next_queue[pos] = neighbor;//Generate the next queue, which could be understood as a queue joining operation
            }
        }
    }
}

//template <typename T>
std::vector<int> cuda_bfs(CSR_graph<double>& input_graph, int source, float* elapsedTime, int max_depth) {
    int V = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();

    int* depth;
    int* edge = input_graph.out_edge;

    int* start = input_graph.out_pointer;
    int* visited;
    
    int* queue, * next_queue;
    int* queue_size, * next_queue_size;

    cudaMallocManaged((void**)&depth, V * sizeof(int));
    cudaMallocManaged((void**)&visited, V * sizeof(int));
    cudaMallocManaged((void**)&queue, V * sizeof(int));
    cudaMallocManaged((void**)&next_queue, V * sizeof(int));
    cudaMallocManaged((void**)&queue_size, sizeof(int));
    cudaMallocManaged((void**)&next_queue_size, sizeof(int));

    for (int i = 0; i < V; i++) {
		depth[i] = max_depth;
		visited[i] = 0;
	}
    depth[source] = 0;


    *queue_size = 1;
    queue[0] = source;
    *next_queue_size = 0;

    int threadsPerBlock = 1024;
    int numBlocks = 0;

    std::vector<int> res(V, max_depth);

    cudaEvent_t start_timer, stop_timer;

    cudaEventCreate(&start_timer);
    cudaEventCreate(&stop_timer);
    cudaEventRecord(start_timer, 0);

    while (*queue_size > 0) {
		numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
		bfs_Relax <<< numBlocks, threadsPerBlock >>> (start, edge, depth, visited, queue, queue_size);
		cudaDeviceSynchronize();

        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Relax kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return res;
        }

		numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;
		bfs_CompactQueue <<< numBlocks, threadsPerBlock >>> (V, next_queue, next_queue_size, visited);
		cudaDeviceSynchronize();

        cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "CompactQueue kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
			return res;
		}
		
        std::swap(queue, next_queue);

		*queue_size = *next_queue_size;
        *next_queue_size = 0;
	}

    cudaEventRecord(stop_timer, 0);
    cudaEventSynchronize(stop_timer);
    cudaEventElapsedTime(elapsedTime, start_timer, stop_timer);

    cudaEventDestroy(start_timer);
    cudaEventDestroy(stop_timer);

    cudaMemcpy(res.data(), depth, V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(depth);
    cudaFree(visited);
    cudaFree(queue);
    cudaFree(next_queue);
    cudaFree(queue_size);
    cudaFree(next_queue_size);

    return res;
}

std::unordered_map<std::string, int> getGPUBFS(std::vector<std::string>& userName, LDBC<double> & graph, CSR_graph<double> &csr_graph){
    std::vector<int> gpuBfsVec = cuda_bfs(csr_graph, graph.bfs_src, 0);
    std::unordered_map<std::string, int> strId2value;

    for(int i = 0; i < gpuBfsVec.size(); ++i){
        // strId2value.emplace(graph.vertex_id_to_str[i], gpuBfsVec[i]);
        strId2value.emplace(userName[i], gpuBfsVec[i]);
    }
    
    return strId2value;
}

/*int main()
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
}*/

/*

nvcc -O3 -std=c++17 -o GPU_BFS.out GPU_BFS.cu
./GPU_BFS.out
rm GPU_BFS.out

*/
