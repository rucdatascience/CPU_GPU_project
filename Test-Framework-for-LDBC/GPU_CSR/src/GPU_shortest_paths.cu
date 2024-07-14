#include <GPU_shortest_paths.cuh>

__device__ __forceinline__ double atomicMinDouble (double * addr, double value) {
    double old;
    old = __longlong_as_double(atomicMin((long long *)addr, __double_as_longlong(value)));
    return old;
}

__global__ void Relax(int* out_pointer, int* out_edge, double* out_edge_weight, double* dis, int* queue, int* queue_size, int* visited) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *queue_size) {
        int v = queue[idx];

        for (int i = out_pointer[v]; i < out_pointer[v + 1]; i++) {
            int new_v = out_edge[i];
            double weight = out_edge_weight[i];

            double new_w = dis[v] + weight;

            double old = atomicMinDouble(&dis[new_v], new_w);

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

void gpu_shortest_paths(CSR_graph<double>& input_graph, int source, std::vector<double>& distance, double max_dis) {
    int V = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();

    double* dis;
    int* out_edge = input_graph.out_edge;
    double* out_edge_weight = input_graph.out_edge_weight;
    int* out_pointer = input_graph.out_pointer;
    int* visited;
    
    int* queue, * next_queue;
    int* queue_size, * next_queue_size;

    cudaMallocManaged((void**)&dis, V * sizeof(double));
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


    *queue_size = 1;
    queue[0] = source;
    *next_queue_size = 0;

    int threadsPerBlock = 1024;
    int numBlocks = 0;

    while (*queue_size > 0) {
		numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
		Relax <<< numBlocks, threadsPerBlock >>> (out_pointer, out_edge, out_edge_weight, dis, queue, queue_size, visited);
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

    cudaMemcpy(distance.data(), dis, V * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dis);
    cudaFree(visited);
    cudaFree(queue);
    cudaFree(next_queue);
    cudaFree(queue_size);
    cudaFree(next_queue_size);

    return;
}

std::map<long long int, double> getGPUSSSP(graph_structure<double> & graph, CSR_graph<double> & csr_graph){
    std::vector<double> gpuSSSPvec(graph.V, 0);
    gpu_shortest_paths(csr_graph, graph.sssp_src, gpuSSSPvec, 10000000000);

    std::map<long long int,   double> strId2value;

    std::vector<long long int> converted_numbers;

    for (const auto& str : graph.vertex_id_to_str) {
        long long int num = std::stoll(str);
        converted_numbers.push_back(num);
    }

    std::sort(converted_numbers.begin(), converted_numbers.end());

	for( int i = 0; i < gpuSSSPvec.size(); ++i){
		strId2value.emplace(converted_numbers[i], gpuSSSPvec[i]);
    }

	// std::string path = "../data/cpu_bfs_75.txt";
	// storeResult(strId2value, path);//ldbc file

    return strId2value;
}

std::vector<std::string> gpu_shortest_paths_v2(graph_structure<double> & graph, CSR_graph<double> &csr_graph){
    std::vector<double> gpuSSSPvec(graph.V, 0);
    gpu_shortest_paths(csr_graph, graph.sssp_src, gpuSSSPvec, 10000000000);

    std::vector<std::string> resultVec;

    for(auto & it : gpuSSSPvec){
		resultVec.push_back(std::to_string(it));
	}

	return resultVec;
}

std::vector<std::pair<std::string, double>> Cuda_SSSP(graph_structure<double>& graph, CSR_graph<double>& csr_graph, std::string src_v, double max_dis) {
    int src_v_id = graph.vertex_str_to_id[src_v];
    std::vector<double> gpuSSSPvec(graph.V, 0);
    gpu_shortest_paths(csr_graph, src_v_id, gpuSSSPvec, max_dis);

    return graph.res_trans_id_val(gpuSSSPvec);
}