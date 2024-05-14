#include "Union-Find.cuh"

/*template std::vector<std::vector<int>> gpu_connected_components<int>(CSR_graph<int>&);
template std::vector<std::vector<int>> gpu_connected_components<float>(CSR_graph<float>&);
template std::vector<std::vector<int>> gpu_connected_components<double>(CSR_graph<double>&);
template std::vector<std::vector<int>> gpu_connected_components<long long>(CSR_graph<long long>&);*/

__device__ int findRoot(int* parent, int i) {
    while (i != parent[i])
        i = parent[i];
    return i;
}

__global__ void Hook(int* parent, int* Start_v, int* End_v, int E) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < E) {
		int u = Start_v[id];
		int v = End_v[id];

        int rootU = findRoot(parent, u);
        int rootV = findRoot(parent, v);

        while (rootU != rootV) {
            int expected = rootU > rootV ? rootU : rootV;
            int desired = rootU < rootV ? rootU : rootV;
            int observed = atomicCAS(&parent[expected], expected, desired);

            if (observed == expected)
                break;

            rootU = findRoot(parent, u);
            rootV = findRoot(parent, v);
        }
    }
}

//template <typename T>
std::vector<std::vector<int>> gpu_connected_components(CSR_graph<double>& input_graph, float* elapsedTime) {
    int N = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();

    // Allocate GPU memory
    int* Start_v;
    int* End_v;
    int* Parent;

    cudaMallocManaged((void**)&Start_v, E * sizeof(int));
    memset(Start_v, 0, E * sizeof(int));
    cudaMallocManaged((void**)&End_v, E * sizeof(int));
    memset(End_v, 0, E * sizeof(int));
    cudaMallocManaged((void**)&Parent, N * sizeof(int));

    // Copy data to GPU
    for (int i = 0; i < N; i++) {
        for (int j = input_graph.OUTs_Neighbor_start_pointers[i]; j < input_graph.OUTs_Neighbor_start_pointers[i + 1]; j++) {
			Start_v[j] = i;
			End_v[j] = input_graph.OUTs_Edges[j];
		}
        Parent[i] = i;
    }

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int threadsPerBlock = 1024;
    int blocksPerGrid = 0;

    blocksPerGrid = (E + threadsPerBlock - 1) / threadsPerBlock;
    Hook<<<blocksPerGrid, threadsPerBlock>>>(Parent, Start_v, End_v, E);
    cudaDeviceSynchronize();
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
		return std::vector<std::vector<int>>();
	}
    // Process components on CPU
    std::vector<std::vector<int>> components;
    std::vector<std::vector<int>> componentLists(N);

    for (int i = 0; i < N; i++) {
        if (Parent[i] != i) {
            int j = i;
            while (Parent[j] != j)
                j = Parent[j];
            Parent[i] = j;
            componentLists[j].push_back(i);
        }
        else
            componentLists[i].push_back(i);
    }

    for (int i = 0; i < N; i++) {
		if (componentLists[i].size() > 0)
			components.push_back(componentLists[i]);
	}

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsedTime, start, stop);
    //printf("Cost time is %f\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free GPU memory
    cudaFree(Start_v);
    cudaFree(End_v);
    cudaFree(Parent);

    return components;
}

/*int main()
{
    graph_v_of_v<int> graph;
    graph.txt_read("example_graph.txt");
    ARRAY_graph<int> arr_graph = graph.toARRAY();
    float sum = 0;
    int it_cnt = 100;
    for (int i = 0; i < it_cnt; i++) {
        gpu_connected_components<int>(arr_graph);
        if (i > 0)
            sum += elapsedTime;
        elapsedTime = 0;
    }
    printf("average cost time is %f ms\n", sum / it_cnt);
    return 0;
}*/

/*

nvcc -O3 -std=c++17 -o Union-Find.out Union-Find.cu
./Union-Find.out
rm Union-Find.out

*/