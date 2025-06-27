#ifndef GPU_BFS_ADJ
#define GPU_BFS_ADJ

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <GPU_adj_list/GPU_adj.hpp>

// #include "util.h"
// #include "header.h"
// #include "barrier.cuh"
// #include "meta_data.cuh"
// #include "mapper.cuh"
// #include "reducer.cuh"
// #include "mapper_enactor.cuh"
// #include "reducer_enactor.cuh"

/*user defined vertex behavior function*/

// __inline__ __host__ __device__ feature_t user_mapper_push // reduce edge_compute_push
// 	(vertex_t src,
// 	 vertex_t dest,
// 	 feature_t level,
// 	 index_t *beg_pos,
// 	 weight_t edge_weight,
// 	 feature_t *vert_status,
// 	 feature_t *vert_status_prev)
// {
// 	return vert_status[src] + edge_weight;
// }

// /*user defined vertex behavior function*/
// __inline__ __host__ __device__ bool vertex_selector_push(vertex_t vert_id,
// 														 feature_t level,
// 														 feature_t *vert_status,
// 														 feature_t *vert_status_prev)
// {
// 	return (vert_status[vert_id] != vert_status_prev[vert_id]);
// }
// __inline__ __device__ bool vertex_selector_best_push(vertex_t vert_id,
// 													 vertex_t width,
// 													 feature_t *vert_status,
// 													 feature_t *vert_status_prev,
// 													 feature_t *one_label,
// 													 volatile feature_t *best,
// 													 feature_t *lb_record,
// 													 feature_t *merge_or_grow,
// 													 feature_t *temp_store)
// {
// 	if (vert_status[vert_id] != vert_status_prev[vert_id] && vert_status[vert_id] - 1 <= 0.5 * (*best))
// 	{
// 		if (lb_record[vert_id] + vert_status[vert_id] <= (*best))
// 			return true;
// 	}
// 	return false;
// }
// /*user defined vertex behavior function*/
// __inline__ __host__ __device__ feature_t user_mapper_pull(vertex_t src,
// 														  vertex_t dest,
// 														  feature_t level,
// 														  index_t *beg_pos,
// 														  weight_t edge_weight,
// 														  feature_t *vert_status,
// 														  feature_t *vert_status_prev)
// {
// 	// return vert_status[src] + edge_weight;
// 	return vert_status[src] + edge_weight;
// }

// /*user defined vertex behavior function*/
// __inline__ __host__ __device__ bool vertex_selector_pull // reduce
// 	(vertex_t vert_id,
// 	 feature_t level,
// 	 feature_t *vert_status,
// 	 feature_t *vert_status_prev)
// {
// 	// return (beg_pos[vert_id] != beg_pos[vert_id + 1]);
// 	// return (vert_status[vert_id] != vert_status_prev[vert_id]);
// 	// return (vert_status[vert_id] != vert_status_prev[vert_id] || vert_status[vert_id] == INFTY);
// 	// return true;
// }

// template<typename T>
std::vector<int> cuda_bfs(GPU_adj<double> &input_graph, int source_vertex, int max_depth = INT_MAX);

std::vector<std::pair<std::string, int>> Cuda_Bfs_adj(graph_structure<double> &graph, GPU_adj<double>& adj_graph, std::string src_v, int min_depth = 0, int max_depth = INT_MAX);

__global__ void bfs_Relax(cuda_vector<int>** edge, int *depth, int *visited, int *queue, int *queue_size) {
    //Relax is performed on each queue node, which traverses all neighboring nodes of that round and relaxes the corresponding distance
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *queue_size) {
        int v = queue[idx];
        int out_edge_size = edge[v]->size();
        cuda_vector<int>& ed = *edge[v];
        for (int i = 0; i < out_edge_size; i++) {
            // Traverse adjacent edges
            int new_v = ed[i];
            int new_depth = depth[v] + 1;

            int old = atomicMin(&depth[new_v], new_depth);//Update distance using atomic operations to avoid conflict
            if (new_depth < old) {
                visited[new_v] = 1;
            }
        }
    }
}

__global__ void bfs_CompactQueue(int V, int *next_queue, int *next_queue_size, int *visited) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V && visited[idx]) {
        //If the node has been accessed in this round, it will be added to the queue for the next round
        int pos = atomicAdd(next_queue_size, 1);
        next_queue[pos] = idx;
        visited[idx] = 0;
    }
}

// template <typename T>
std::vector<int> cuda_bfs(GPU_adj<double> &input_graph, int source, int max_depth) {
/*     The GPU code for breadth first search uses queues to traverse the graph and record depth,
     which is also used to prevent duplicate traversal */
    int V = input_graph.V;

    int *depth;
    auto edge = input_graph.out_edge();

    int *visited;

    int *queue, *next_queue;
    int *queue_size, *next_queue_size;

    cudaMallocManaged((void **)&depth, V * sizeof(int));
    cudaMallocManaged((void **)&visited, V * sizeof(int));
    cudaMallocManaged((void **)&queue, V * sizeof(int));
    cudaMallocManaged((void **)&next_queue, V * sizeof(int));
    cudaMallocManaged((void **)&queue_size, sizeof(int));
    cudaMallocManaged((void **)&next_queue_size, sizeof(int));
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

    *queue_size = 1; // At first, there was only the root node in the queue
    queue[0] = source;
    *next_queue_size = 0;

    int threadsPerBlock = 1024;
    int numBlocks = 0;
    int QBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;
    std::vector<int> res(V, max_depth);

    while (*queue_size > 0) {
        numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
        bfs_Relax<<<numBlocks, threadsPerBlock>>>(edge, depth, visited, queue, queue_size);
        cudaDeviceSynchronize();

        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Relax kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return res;
        }

        bfs_CompactQueue<<<QBlocks, threadsPerBlock>>>(V, next_queue, next_queue_size, visited);
        cudaDeviceSynchronize();

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "CompactQueue kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return res;
        }

        std::swap(queue, next_queue);
        *queue_size = *next_queue_size;
        *next_queue_size = 0;
        /*  After each round of updates, exchange pointers between the new and old queues,
         using the new queue as the traversal queue for the next round and the old queue as the new queue for the next round */
    }
    cudaMemcpy(res.data(), depth, V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(depth);
    cudaFree(visited);
    cudaFree(queue);
    cudaFree(next_queue);
    cudaFree(queue_size);
    cudaFree(next_queue_size);
    return res;
}

// __device__ cb_reducer vert_selector_push_d = vertex_selector_push;
// __device__ cb_reducer vert_selector_pull_d = vertex_selector_pull;
// __device__ cb_mapper vert_behave_push_d = user_mapper_push;
// __device__ cb_mapper vert_behave_pull_d = user_mapper_pull;
// __device__ best_reducer vertex_selector_best_push_d = vertex_selector_best_push;

std::vector<std::pair<std::string, int>> Cuda_Bfs_adj(graph_structure<double> &graph, GPU_adj<double> &adj_graph, std::string src_v, int min_depth, int max_depth) {
    int V = graph.V;
    int *dep = new int[V];
    int src_v_id = graph.vertex_str_to_id[src_v];
    std::vector<int> gpuBfsVec = cuda_bfs(adj_graph, src_v_id, max_depth);

    // cb_reducer vert_selector_push_h;
	// cb_reducer vert_selector_pull_h;
	// best_reducer vert_selector_push_h_best;
	// cudaMemcpyFromSymbol(&vert_selector_push_h_best, vertex_selector_best_push_d, sizeof(cb_reducer));
	// cudaMemcpyFromSymbol(&vert_selector_push_h, vert_selector_push_d, sizeof(cb_reducer));
	// cudaMemcpyFromSymbol(&vert_selector_pull_h, vert_selector_pull_d, sizeof(cb_reducer));

    // cb_mapper vert_behave_push_h;
	// cb_mapper vert_behave_pull_h;
	// cudaMemcpyFromSymbol(&vert_behave_push_h, vert_behave_push_d, sizeof(cb_reducer));
	// cudaMemcpyFromSymbol(&vert_behave_pull_h, vert_behave_pull_d, sizeof(cb_reducer));

    // int blk_size = 512;
    // meta_data mdata(graph.V, 0, 1);
    // Barrier global_barrier(BLKS_NUM);
    // mapper compute_mapper(adj_graph, mdata, vert_behave_push_h, vert_behave_pull_h, 1);
	// reducer worklist_gather(adj_graph, mdata, vert_selector_push_h, vert_selector_pull_h, vert_selector_push_h_best, 1);
	// feature_t *level, *level_h;
    // cudaMalloc((void **)&level, 10 * sizeof(feature_t));
	// cudaMallocHost((void **)&level_h, 10 * sizeof(feature_t));
    // cudaMemset(level, 0, sizeof(feature_t));

    // std::fill(dep, dep + V, max_depth);
    // cudaMemcpy(mdata.vert_status_prev, dep, V * sizeof(int), cudaMemcpyHostToDevice);
    // dep[src_v_id] = 0;
	// cudaMemcpy(mdata.vert_status, dep, V * sizeof(int), cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();

    // balanced_push(blk_size, level, adj_graph, mdata, compute_mapper, worklist_gather, global_barrier);
    
    // std::vector<int> gpuBfsVec(V, max_depth);
    // cudaMemcpy(gpuBfsVec.data(), mdata.vert_status, V * sizeof(int), cudaMemcpyDeviceToHost);
    
    return graph.res_trans_id_val(gpuBfsVec);
}

#endif