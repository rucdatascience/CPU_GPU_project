#include <GPU_connected_components.cuh>

/*template std::vector<std::vector<int>> gpu_connected_components<int>(CSR_graph<int>&);
template std::vector<std::vector<int>> gpu_connected_components<float>(CSR_graph<float>&);
template std::vector<std::vector<int>> gpu_connected_components<double>(CSR_graph<double>&);
template std::vector<std::vector<int>> gpu_connected_components<long long>(CSR_graph<long long>&);*/

__device__ int findRoot(int* parent, int i) {
    //Recursively searching for the ancestor of node i
    while (i != parent[i])
        i = parent[i];//
    return i;
}

__global__ void Hook(int* parent, int* Start_v, int* End_v, int E) {
    //Merge operations on each edge
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //Calculate thread ID
    if (id < E) {
		int u = Start_v[id];
		int v = End_v[id];
        //u,v are the starting and ending points of the edge
        int rootU = findRoot(parent, u);
        int rootV = findRoot(parent, v);
        //Obtain Root Node
        while (rootU != rootV) {
            int expected = rootU > rootV ? rootU : rootV;
            int desired = rootU < rootV ? rootU : rootV;
           //During multi-core operations, the root node may be manipulated by other threads, so locking is necessary for the operation
            int observed = atomicCAS(&parent[expected], expected, desired);
            /*
            compare and swap
            int atomicCAS(int* address, int compare, int val);
            Check if the address and compare are the same. If they are the same, enter address as desired. Otherwise, no action will be taken
            observed = parent[expected]

            */
            
          
            if (observed == expected)//If the observed values are correct and the merge operation is successful, exit the loop
                break;
            //If the observed value has been modified, the modified new root node needs to be obtained
            rootU = findRoot(parent, u);
            rootV = findRoot(parent, v);
        }
    }
}

//template <typename T>
std::vector<std::vector<int>> gpu_connected_components(CSR_graph<double>& input_graph, float* elapsedTime) {
    int N = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();
    //Number of nodes and edges
    
    int* Start_v;
    int* End_v;
    int* Parent;
    // Allocate GPU memory
    cudaMallocManaged((void**)&Start_v, E * sizeof(int));
    memset(Start_v, 0, E * sizeof(int));
    cudaMallocManaged((void**)&End_v, E * sizeof(int));
    memset(End_v, 0, E * sizeof(int));
    cudaMallocManaged((void**)&Parent, N * sizeof(int));
    //Forming an edge list
    // Copy data to GPU
    for (int i = 0; i < N; i++) {
        for (int j = input_graph.OUTs_Neighbor_start_pointers[i]; j < input_graph.OUTs_Neighbor_start_pointers[i + 1]; j++) {
			Start_v[j] = i;
			End_v[j] = input_graph.OUTs_Edges[j];
		}
        Parent[i] = i;//initialization
    }

    cudaEvent_t start, stop;
    //Used to create events and measure the time of GPU operations.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int threadsPerBlock = 1024;
    int blocksPerGrid = 0;
    //Disperse E operations on threads
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
    //Using a linked list to record connected components
    for (int i = 0; i < N; i++) {
        if (Parent[i] != i) {
            //If it is not the root node, add the node to the linked list of the root node it belongs to
            int j = i;
            while (Parent[j] != j)
                j = Parent[j];
            Parent[i] = j;
            componentLists[j].push_back(i);
        }
        else  //The root node is directly added to the root node linked list
            componentLists[i].push_back(i);
    }

    for (int i = 0; i < N; i++) {
		if (componentLists[i].size() > 0)
            //Filter non empty connected components
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
