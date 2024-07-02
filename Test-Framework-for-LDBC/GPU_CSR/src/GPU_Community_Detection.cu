#include <GPU_Community_Detection.cuh>
static int CD_ITERATION;
static int *new_labels, *labels; // two array to prop_labels the labels of nodes
static int *all_pointer,*all_edge,*prop_labels;
static int N;
static long long E;

// init the CSR information depending on the graph_structure

__global__ void LabelPropagation(int *all_pointer, int *prop_labels, int *labels, int *all_edge, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize all labels at once with GPU.Initially, each vertex v is assigned a unique label which matches its identifier.

    if (tid >= 0 && tid < N)
    {
        for (int c = all_pointer[tid]; c < all_pointer[tid + 1]; c++)
        {
            prop_labels[c] = labels[all_edge[c]];
        }
    }
}
__global__ void Label_init(int *labels, int *all_pointer, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize all labels at once with GPU.Initially, each vertex v is assigned a unique label which matches its identifier.

    if (tid >= 0 && tid < N)
    {
        labels[tid] = tid;
    }

}

// each thread is responsible for one vertex
// every segmentation are sorted
// count Frequency from the start in the global_space_for_label to the end in the global_space_for_label
// the new labels are stroed in the new_labels_gpu
__global__ void Get_New_Label(int *all_pointer,int *prop_labels, int *new_labels, int N)
{ // Use GPU to propagate all labels at the same time.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 0 && tid < N)
    {

        thrust::sort(thrust::device,prop_labels+all_pointer[tid], prop_labels+all_pointer[tid + 1]);
        int maxlabel = prop_labels[all_pointer[tid]], maxcount = 0;
        for (int c = all_pointer[tid], last_label = prop_labels[all_pointer[tid]], last_count = 0; c < all_pointer[tid + 1]; c++)
        {
            if (prop_labels[c] == last_label)
            {
                last_count++;
                if (last_count > maxcount)
                {
                    maxcount = last_count;
                    maxlabel = last_label;
                }
            }
            else
            {
                last_label = prop_labels[c];
                last_count = 1;
            }
        }
        new_labels[tid] = maxlabel;
    }
}

void CDLP_GPU(LDBC<double> &graph, CSR_graph<double> &input_graph, std::vector<string> &res)
{

    dim3 init_label_block((N + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1);
    dim3 init_label_thread(CD_THREAD_PER_BLOCK, 1, 1);
    all_edge = input_graph.all_edge, all_pointer = input_graph.all_pointer;
    CD_ITERATION = graph.cdlp_max_its;
    N = graph.size();
    E = input_graph.E_all;
    cout << N << " and " << E << endl;
    cudaMallocManaged(&new_labels, N * sizeof(int));
    cudaMallocManaged(&labels, N * sizeof(int));
    cudaMallocManaged(&prop_labels, E * sizeof(int));
    //cudaMallocManaged(&flags, E * sizeof(int));
    Label_init<<<init_label_block, init_label_thread>>>(labels,all_pointer,N);
    // thrust::sequence(labels, labels + N, 0, 1);

    cout << endl;
    int it = 0;
    while (it < CD_ITERATION)
    {
        /* thrust::stable_sort_by_key(thrust::device, prop_labels.begin(), prop_labels.end(), flags);
        thrust::stable_sort_by_key(thrust::device, flags, flags + E, prop_labels); */
        cout << "round " << it << endl;
        // Calculate the neighbor label array for each vertex
        LabelPropagation<<<init_label_block, init_label_thread>>>(all_pointer, prop_labels, labels, all_edge, N);
        cudaDeviceSynchronize();
        Get_New_Label<<<init_label_block, init_label_thread>>>(all_pointer, prop_labels, new_labels, N);
        cudaDeviceSynchronize();
        it++;
        std::swap(labels, new_labels);
    }
    int *gpu_res = new int[N];
    cudaMemcpy(gpu_res, labels, N * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < GRAPHSIZE; ++i){
    //     cout<<"the gpu_res is:"<<gpu_res[i]<<endl;
    // }
    for (int i = 0; i < N; i++)
    {
        res[i] = graph.vertex_id_to_str[gpu_res[i]];
    }
    for (int i = 0; i < 100; i++)
    {
        cout << res[i] << " ";
    }
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << endl;
        exit(EXIT_FAILURE);
    }
}
/*     int num_elements = in_pointer[tid + 1] - in_pointer[tid] + out_pointer[tid + 1] - out_pointer[tid];
    thrust::device_vector<int> d_vec(num_elements), d_ones(num_elements), output_key(N), output_freq(N);
    thrust::fill(d_ones.begin(), d_ones.end(), 1);
    for (int i = in_pointer[tid]; i < in_pointer[tid + 1]; ++i)
    {
        d_vec.push_back(labels[in_edge[i]]);
    }
    for (int i = out_pointer[tid]; i < out_pointer[tid + 1]; ++i)
    {
        d_vec.push_back(labels[in_edge[i]]);
    }
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
    new_end = thrust::reduce_by_key(d_vec.begin(), d_vec.end(), d_ones.begin(), output_key.begin(), output_freq.begin());
    int num_keys = new_end.first - output_key.begin();
    thrust::device_vector<int>::iterator iter = thrust::max_element(output_freq.begin(), output_freq.end());

    unsigned int index = iter - output_freq.begin();

    new_labels[tid] = output_key[index];

     */

std::map<long long int, string> getGPUCDLP(LDBC<double> & graph, CSR_graph<double> & csr_graph){
    std::vector<string> ans_gpu(graph.size());
    CDLP_GPU(graph, csr_graph,ans_gpu);

    std::map<long long int, string> strId2value;

    std::vector<long long int> converted_numbers;

    for (const auto& str : graph.vertex_id_to_str) {
        long long int num = std::stoll(str);
        converted_numbers.push_back(num);
    }

    std::sort(converted_numbers.begin(), converted_numbers.end());

    for(int i = 0; i < ans_gpu.size(); ++i){
        strId2value.emplace(converted_numbers[i], ans_gpu[i]);
    }

    // std::string path = "../data/cpu_bfs_75.txt";
	// storeResult(strId2value, path);//ldbc file
    
    return strId2value;
}
