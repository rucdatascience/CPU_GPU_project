#ifndef GPU_BFS
#define GPU_BFS

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include <GPU_csr/GPU_csr.hpp>

#include <map>

__global__ void bfs_kernel(int* edges, int* start, int* visited, int* queue, int* next_queue, int* queue_size, int* next_queue_size, int max_depth);

//template<typename T>
std::vector<int> cuda_bfs(CSR_graph<double>& input_graph, int source_vertex, int max_depth = INT_MAX);

std::vector<std::pair<std::string, int>> Cuda_Bfs(graph_structure<double>& graph, CSR_graph<double>& csr_graph, std::string src_v, int min_depth = 0, int max_depth = INT_MAX);


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
std::vector<int> cuda_bfs(CSR_graph<double>& input_graph, int source, int max_depth) {
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

    cudaDeviceSynchronize();

    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return std::vector<int>();
    }

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

    cudaMemcpy(res.data(), depth, V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(depth);
    cudaFree(visited);
    cudaFree(queue);
    cudaFree(next_queue);
    cudaFree(queue_size);
    cudaFree(next_queue_size);

    return res;
}

std::vector<std::pair<std::string, int>> Cuda_Bfs(graph_structure<double>& graph, CSR_graph<double>& csr_graph, std::string src_v, int min_depth, int max_depth) {
    int src_v_id = graph.vertex_str_to_id[src_v];
    std::vector<int> gpuBfsVec = cuda_bfs(csr_graph, src_v_id, max_depth);

    return graph.res_trans_id_val(gpuBfsVec);
}

#endif