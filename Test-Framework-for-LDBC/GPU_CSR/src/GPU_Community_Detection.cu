#include <GPU_Community_Detection.cuh>
#include <cub/cub.cuh>
static int CD_ITERATION;
static int *new_labels, *labels; // two array to prop_labels the labels of nodes
static int *all_pointer, *all_edge, *prop_labels, *counts, *new_prop_labels;
static int N;
static long long E;

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
__global__ void Get_New_Label(int *all_pointer, int *prop_labels, int *new_labels, int *labels, int *counts, int N)
{ // Use GPU to propagate all labels at the same time.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 0 && tid < N)
    {
        /*
        violence test

        for (int i = all_pointer[tid]; i < all_pointer[tid+1]; i++)
         {
             counts[i] = 0;

         }

         for (int i = all_pointer[tid]; i < all_pointer[tid + 1]; i++)
         {
             for (int j = i+1; j < all_pointer[tid + 1]; j++)
             {
                 if (prop_labels[j]==prop_labels[i])
                 {
                     counts[j]++;
                 }

             }

 int maxlabel = all_pointer[tid];
         for (int i = all_pointer[tid]; i < all_pointer[tid+1]; i++)
         {
             if (counts[i]>counts[maxlabel])
             {
                 maxlabel = i;
             }
             else if (counts[i]==counts[maxlabel])
             {
                 if (prop_labels[i]<prop_labels[maxlabel])
                 {
                     maxlabel = i;
                 }
             }


         }
     new_labels[tid] = prop_labels[maxlabel];
         }  */

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
    N = graph.size();
    dim3 init_label_block((N + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1);
    dim3 init_label_thread(CD_THREAD_PER_BLOCK, 1, 1);
    all_edge = input_graph.all_edge, all_pointer = input_graph.all_pointer;
    CD_ITERATION = graph.cdlp_max_its;
    E = input_graph.E_all;

    cout << N << " and " << E << endl;
    cudaMallocManaged(&new_labels, N * sizeof(int));
    cudaMallocManaged(&labels, N * sizeof(int));
    cudaMallocManaged(&prop_labels, E * sizeof(int));
    cudaMallocManaged(&new_prop_labels, E * sizeof(int));
    cudaMallocManaged(&counts, E * sizeof(int));
    // cudaMallocManaged(&flags, E * sizeof(int));
    Label_init<<<init_label_block, init_label_thread>>>(labels, all_pointer, N);
    // thrust::sequence(labels, labels + N, 0, 1);

    int it = 0;
    while (it < CD_ITERATION)
    {
        /* thrust::stable_sort_by_key(thrust::device, prop_labels.begin(), prop_labels.end(), flags);
        thrust::stable_sort_by_key(thrust::device, flags, flags + E, prop_labels); */
        // Calculate the neighbor label array for each vertex
        LabelPropagation<<<init_label_block, init_label_thread>>>(all_pointer, prop_labels, labels, all_edge, N);

        cudaDeviceSynchronize();

        // Determine temporary device storage requirements
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceSegmentedSort::SortKeys(
            d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels,
            N, N, all_pointer, all_pointer + 1);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Run sorting operation
        cub::DeviceSegmentedSort::SortKeys(
            d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels,
            N, N, all_pointer, all_pointer + 1);
        std::swap(new_prop_labels, prop_labels);
        Get_New_Label<<<init_label_block, init_label_thread>>>(all_pointer, prop_labels, new_labels, labels, counts, N);
        cudaDeviceSynchronize();
        it++;
        std::swap(labels, new_labels);

        cout << "round " << it << " finish" << endl;
    }
    cudaFree(prop_labels);
    cudaFree(new_labels);
    printf("\n result:");

    for (int i = 0; i < N; i++)
    {
        res[i] = graph.vertex_id_to_str[labels[i]];
    }

    for (int i = 0; i < 100; i++)
    {
        cout << res[i] << " ";
    }
    cout << endl;
    cudaFree(labels);
    cudaFree(new_prop_labels);
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

std::map<long long int, string> getGPUCDLP(LDBC<double> &graph, CSR_graph<double> &csr_graph)
{
    std::vector<string> ans_gpu(graph.size());
    CDLP_GPU(graph, csr_graph, ans_gpu);

    std::map<long long int, string> strId2value;

    std::vector<long long int> converted_numbers;

    for (const auto &str : graph.vertex_id_to_str)
    {
        long long int num = std::stoll(str);
        converted_numbers.push_back(num);
    }

    std::sort(converted_numbers.begin(), converted_numbers.end());

    for (int i = 0; i < ans_gpu.size(); ++i)
    {
        strId2value.emplace(converted_numbers[i], ans_gpu[i]);
    }

    // std::string path = "../data/cpu_bfs_75.txt";
    // storeResult(strId2value, path);//ldbc file

    return strId2value;
}
