#include "Workfront-Sweep.cuh"

float elapsedTime = 0.0;

__global__ void Relax(int* offsets, int* edges, int* weights, int* dis, int* queue, int* queue_size, int* visited) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *queue_size) {
        int v = queue[idx];

        for (int i = offsets[v]; i < offsets[v + 1]; i++) {
            int new_v = edges[i];
            int weight = weights[i];

            int new_w = dis[v] + weight;

            int old = atomicMin(&dis[new_v], new_w);

            if (old <= new_w)
				continue;

            atomicExch(&visited[new_v], 1);
        }
    }
}

__global__ void CompactQueue(int V, int* next_queue, int* next_queue_size, int* visited) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V && visited[idx]) {
        int pos = atomicAdd(next_queue_size, 1);
        next_queue[pos] = idx;
        visited[idx] = 0;
    }
}

void Workfront_Sweep(ARRAY_graph<int>& input_graph, int source, std::vector<int>& distance, int max_dis) {
    int V = input_graph.Neighbor_start_pointers.size() - 1;
    int E = input_graph.Edges.size();

    int* dis;
    int* edges;
    int* weights;
    int* offsets;
    int* visited;
    
    int* queue, * next_queue;
    int* queue_size, * next_queue_size;

    cudaMallocManaged((void**)&dis, V * sizeof(int));
    cudaMallocManaged((void**)&edges, E * sizeof(int));
    cudaMallocManaged((void**)&weights, E * sizeof(int));
    cudaMallocManaged((void**)&offsets, (V + 1) * sizeof(int));
    cudaMallocManaged((void**)&visited, V * sizeof(int));
    cudaMallocManaged((void**)&queue, V * sizeof(int));
    cudaMallocManaged((void**)&next_queue, V * sizeof(int));
    cudaMallocManaged((void**)&queue_size, sizeof(int));
    cudaMallocManaged((void**)&next_queue_size, sizeof(int));

    for (int i = 0; i < V; i++) {
		dis[i] = max_dis;
		visited[i] = 0;
	}
    dis[source] = 0;
    cudaMemcpy(offsets, input_graph.Neighbor_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edges, input_graph.Edges.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(weights, input_graph.Edge_weights.data(), E * sizeof(int), cudaMemcpyHostToDevice);

    *queue_size = 1;
    queue[0] = source;
    *next_queue_size = 0;

    int threadsPerBlock = 1024;
    int numBlocks = 0;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (*queue_size > 0) {
		numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
		Relax <<< numBlocks, threadsPerBlock >>> (offsets, edges, weights, dis, queue, queue_size, visited);
		cudaDeviceSynchronize();

        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Relax kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

		numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;
		CompactQueue <<< numBlocks, threadsPerBlock >>> (V, next_queue, next_queue_size, visited);
		cudaDeviceSynchronize();

        cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "CompactQueue kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
			return;
		}
		
        std::swap(queue, next_queue);

		*queue_size = *next_queue_size;
        *next_queue_size = 0;
	}

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(distance.data(), dis, V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dis);
    cudaFree(edges);
    cudaFree(weights);
    cudaFree(offsets);
    cudaFree(visited);
    cudaFree(queue);
	cudaFree(next_queue);
	cudaFree(queue_size);
	cudaFree(next_queue_size);

    return;
}

int main()
{
    std::string file_path;
    std::cout << "Please input the file path of the graph: ";
    std::cin >> file_path;
    graph_v_of_v<int> graph;
    graph.txt_read(file_path);
    ARRAY_graph<int> arr_graph = graph.toARRAY();
    float sum = 0;
    int V = arr_graph.Neighbor_start_pointers.size() - 1;
    std::vector<int> distance(V, 0);
    Workfront_Sweep(arr_graph, 0, distance, 1000000);
    for (int i = 0; i < V; i++) {
        Workfront_Sweep(arr_graph, i, distance, 1000000);
        sum += elapsedTime;
        elapsedTime = 0;
    }
    printf("GPU average time: %f ms\n", sum / V);
    return 0;
}

/*

nvcc -O3 -std=c++17 -o Workfront-Sweep.out Workfront-Sweep.cu
./Workfront-Sweep.out
rm Workfront-Sweep.out

*/