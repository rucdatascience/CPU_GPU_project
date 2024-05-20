#include <GPU_Community_Detection.cuh>
using namespace std;

static int CD_GRAPHSIZE;
static int CD_ITERATION;
static int CD_SET_THREAD;
static vector<int> outs_ptr, ins_ptr, outs_neighbor, ins_neighbor;

static int *outs_ptr_gpu, *ins_ptr_gpu;
static int *outs_neighbor_gpu, *ins_neighbor_gpu;
static int *new_labels_gpu, *labels_gpu;
static int *global_space_for_label_count;
template <typename T>
void pre_set(graph_structure<T> &graph, int &CD_GRAPHSIZE)
{

    CSR_graph<T> ARRAY_graph;
    ARRAY_graph = graph.toCSR();

    CD_ITERATION = graph.cdlp_max_its;
    CD_GRAPHSIZE = ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;
    CD_SET_THREAD = 100;

    outs_ptr.resize(CD_GRAPHSIZE + 1);
    outs_ptr = ARRAY_graph.OUTs_Neighbor_start_pointers;
    ins_ptr.resize(CD_GRAPHSIZE + 1);
    ins_ptr = ARRAY_graph.INs_Neighbor_start_pointers;

    outs_neighbor = ARRAY_graph.OUTs_Edges;
    ins_neighbor = ARRAY_graph.INs_Edges;
}

__global__ void init_label(int *labels_gpu, int CD_GRAPHSIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < CD_GRAPHSIZE)
    {
        labels_gpu[tid] = tid;
    }
}

__global__ void init_global_space(int *global_space_for_label_count, int CD_GRAPHSIZE, int CD_SET_THREAD)
{
    int tid = threadIdx.x;
    if (tid >= CD_SET_THREAD)
    {
        return;
    }
    int start = tid * CD_GRAPHSIZE;
    for (int i = 0; i < CD_GRAPHSIZE; ++i)
    {
        global_space_for_label_count[start + i] = 0;
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

void checkDeviceProperties()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Device name: " << prop.name << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;
    cout << "Max blocks per dimension: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
    cout << "Max shared memory per block: " << prop.sharedMemPerBlock << " bytes" << endl;
    cout << "Total global memory: " << prop.totalGlobalMem << " bytes" << endl;
}

void get_size()
{
    size_t freeMem = 0;
    size_t totalMem = 0;

    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);

    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Free memory: " << freeMem << " Byte" << std::endl;
    std::cout << "Total memory: " << totalMem << " Byte" << std::endl;
    size_t a = CD_GRAPHSIZE * 8;
    cout << "space for single thread  : " << a << endl;
    cout << "max thread num : " << freeMem / a << endl;
    size_t t = (size_t)(20LL * (1LL << 30)) / a;

    cout << "if use 20GB : " << t << endl;
    return;
}

__global__ void LPA(int *global_space_for_label_count, int *outs_ptr_gpu, int *ins_ptr_gpu, int *outs_neighbor_gpu, int *ins_neighbor_gpu, int *labels_gpu, int *new_labels_gpu, int CD_SET_THREAD, int CD_GRAPHSIZE, int epoch_iteration)
{
    int tid = threadIdx.x;
    if (tid >= CD_SET_THREAD)
        return;
    int ver = epoch_iteration * CD_SET_THREAD + tid;
    if (ver >= CD_GRAPHSIZE)
        return;
    int *segment_start = global_space_for_label_count + (epoch_iteration * CD_SET_THREAD + tid);
    int maxCount = -1, maxLabel = CD_GRAPHSIZE;
    int outs_start = outs_neighbor_gpu[ver], outs_end = outs_neighbor_gpu[ver + 1];
    int ins_start = ins_neighbor_gpu[ver], ins_end = ins_neighbor_gpu[ver + 1];
    int total = (outs_end - outs_start + ins_end - ins_start) / 2;
    bool find = false;
    for (int i = outs_start; i < outs_end; ++i)
    {
        int temp_neighbor = outs_neighbor_gpu[i];
        int temp_label = labels_gpu[temp_neighbor];
        segment_start[temp_label]++;
        if (segment_start[temp_label] > maxCount)
        {
            maxCount = segment_start[temp_label];
            maxLabel = temp_label;
        }
        else if (segment_start[temp_label] == maxCount && temp_label < maxLabel)
        {
            maxLabel = temp_label;
        }
        if (maxCount > total)
        {
            new_labels_gpu[ver] = maxLabel;
            find = true;
            break;
        }
    }
    if (!find)
    {
        for (int i = ins_start; i < ins_end; ++i)
        {
            int temp_neighbor = ins_neighbor_gpu[i];
            int temp_label = labels_gpu[temp_neighbor];
            segment_start[temp_label]++;
            if (segment_start[temp_label] > maxCount)
            {
                maxCount = segment_start[temp_label];
                maxLabel = temp_label;
            }
            else if (segment_start[temp_label] == maxCount && temp_label < maxLabel)
            {
                maxLabel = temp_label;
            }
            if (maxCount > total)
            {
                new_labels_gpu[ver] = maxLabel;
                break;
            }
        }
    }
    new_labels_gpu[ver] = maxLabel;
    
    for(int i=outs_start;i<outs_end;++i){
        int temp_neighbor = outs_neighbor_gpu[i];
        int temp_label = labels_gpu[temp_neighbor];
        segment_start[temp_label]=0;
    }
    for(int i=ins_start;i<ins_end;++i){
        int temp_neighbor = ins_neighbor_gpu[i];
        int temp_label = labels_gpu[temp_neighbor];
        segment_start[temp_label]=0;
    }
    return;
}

int Community_Detection(graph_structure<double> &graph, float *elapsedTime)
{
    pre_set(graph, CD_GRAPHSIZE);

    dim3 init_label_block((CD_GRAPHSIZE + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1);
    dim3 init_label_thread(CD_THREAD_PER_BLOCK, 1, 1);
    dim3 LPA_block(1, 1, 1);
    dim3 LPA_thread(CD_SET_THREAD, 1, 1);
    cudaMalloc(&outs_ptr_gpu, (CD_GRAPHSIZE + 1) * sizeof(int));
    cudaMalloc(&ins_ptr_gpu, (CD_GRAPHSIZE + 1) * sizeof(int));
    cudaMalloc(&labels_gpu, CD_GRAPHSIZE * sizeof(int));
    cudaMalloc(&new_labels_gpu, CD_GRAPHSIZE * sizeof(int));
    cudaMalloc(&outs_neighbor_gpu, outs_neighbor.size() * sizeof(int));
    cudaMalloc(&ins_neighbor_gpu, ins_neighbor.size() * sizeof(int));
    cudaMalloc(&global_space_for_label_count, CD_SET_THREAD * CD_GRAPHSIZE * sizeof(int));
    cudaError_t err = cudaMalloc(&global_space_for_label_count, CD_SET_THREAD * CD_GRAPHSIZE * sizeof(int));
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    cudaMemcpy(outs_ptr_gpu, outs_ptr.data(), outs_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ins_ptr_gpu, ins_ptr.data(), ins_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(outs_neighbor_gpu, outs_neighbor.data(), outs_neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ins_neighbor_gpu, ins_neighbor.data(), ins_neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);

    // checkDeviceProperties();
    // get_size();

    init_label<<<init_label_block, init_label_thread>>>(labels_gpu, CD_GRAPHSIZE);
    cudaDeviceSynchronize();
    init_global_space<<<LPA_block, LPA_thread>>>(global_space_for_label_count, CD_GRAPHSIZE, CD_SET_THREAD);
    cudaDeviceSynchronize();

    cudaEvent_t GPUstart, GPUstop;
    cudaEventCreate(&GPUstart);
    cudaEventCreate(&GPUstop);
    cudaEventRecord(GPUstart, 0);

    int it = 0;
    cout<<"total epoch_iteration : "<<(CD_GRAPHSIZE + CD_SET_THREAD - 1) / CD_SET_THREAD<<endl;
    while (it < 1)
    {
        if (it % 2 == 0)
        {
            for (int i = 0; i < (CD_GRAPHSIZE + CD_SET_THREAD - 1) / CD_SET_THREAD; i++)
            {
                cout<<"epoch_iteration : "<<i <<endl;
                LPA<<<LPA_block, LPA_thread>>>(global_space_for_label_count, outs_ptr_gpu, ins_ptr_gpu, outs_neighbor_gpu, ins_neighbor_gpu, labels_gpu, new_labels_gpu, CD_SET_THREAD, CD_GRAPHSIZE, i);
                err = cudaDeviceSynchronize();
                //checkCudaError(err, "cudaDeviceSynchronize after LPA");
            }
        }
        else
        {
            for (int i = 0; i < (CD_GRAPHSIZE + CD_SET_THREAD - 1) / CD_SET_THREAD; i++)
            {
                LPA<<<LPA_block, LPA_thread>>>(global_space_for_label_count, outs_ptr_gpu, ins_ptr_gpu, outs_neighbor_gpu, ins_neighbor_gpu, new_labels_gpu, labels_gpu, CD_SET_THREAD, CD_GRAPHSIZE, i);
                err = cudaDeviceSynchronize();
                checkCudaError(err, "cudaDeviceSynchronize after LPA");
            }
        }

        it++;
    }

    cudaEventRecord(GPUstop, 0);
    cudaEventSynchronize(GPUstop);

    cudaEventElapsedTime(elapsedTime, GPUstart, GPUstop);
    cudaEventDestroy(GPUstart);
    cudaEventDestroy(GPUstop);

    cudaFree(outs_ptr_gpu);
    cudaFree(ins_ptr_gpu);
    cudaFree(labels_gpu);
    cudaFree(outs_neighbor_gpu);
    cudaFree(ins_neighbor_gpu);
    cudaFree(new_labels_gpu);

    return 0;
}
