#include"LPA_CUDA_BLOCK.cuh"
using namespace std;

int GRAPHSIZE;
vector<int> row_ptr, col_indices;
vector<int> neighbor;
int* blockPerVer;
int* row_ptr_gpu, * col_indices_gpu;
int* labels_gpu, * neighbor_gpu;
int* reduce_label, * reduce_label_count;
int* updating;
int* edge_size;

void make_csr(graph_structure &graph, int& GRAPHSIZE)
{
    GRAPHSIZE = graph.ADJs.size();
    cout<<GRAPHSIZE<<endl;
    row_ptr.resize(GRAPHSIZE + 1);
    row_ptr[0] = 0;
    for (int i = 0; i < GRAPHSIZE; i++)
    {
        for (auto& edge : graph.ADJs[i])
        {
            int neighbor_vertex = edge.first;
            neighbor.push_back(neighbor_vertex);
            col_indices.push_back(neighbor_vertex);
        }
        row_ptr[i + 1] = row_ptr[i] + graph.ADJs[i].size();
    }
    cout << "CSR 矩阵已创建" << endl;
}


__global__ void init_label(int* labels_gpu,int GRAPHSIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < GRAPHSIZE)
    {
        labels_gpu[tid] = tid;
    }
}

__global__ void LPA(int* row_ptr_gpu, int* labels_gpu, int* neighbor_gpu, int* reduce_label, int* reduce_label_count,int GRAPHSIZE,int BLOCK_PER_VER)
{
    extern __shared__ int label_counts[];
    extern __shared__ int label[];
    int ver = blockIdx.x / BLOCK_PER_VER;
    int tid = (blockIdx.x % BLOCK_PER_VER) * blockDim.x + threadIdx.x;
    int segment_order = blockIdx.x % BLOCK_PER_VER;
    int stid = threadIdx.x;
    if (stid == ver)
    {
        label_counts[stid] = 1;
    }
    else
    {
        label_counts[stid] = 0;
    }
    label[stid] = tid;

    __syncthreads();

    int start = row_ptr_gpu[ver], end = row_ptr_gpu[ver + 1];
    if (tid >= end - start)
    {
        return;
    }
    int neighbor_label = labels_gpu[neighbor_gpu[start + tid]];
    if (neighbor_label >= segment_order * THREAD_PER_BLOCK && neighbor_label < (segment_order + 1) * THREAD_PER_BLOCK)
        atomicAdd(&label_counts[neighbor_label - segment_order * THREAD_PER_BLOCK], 1);

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (label_counts[tid] < label_counts[tid + s])
            {
                label_counts[tid] = label_counts[tid + s];
                label[tid] = label_counts[tid + s];
            }
            else if (label_counts[tid] == label_counts[tid + s] && label[tid] > label_counts[tid + s])
            {
                label[tid] = label_counts[tid + s];
            }
        }
        __syncthreads();
    }
    reduce_label_count[blockIdx.x] = label_counts[0];
    reduce_label[blockIdx.x] = label[0];
    return;
}

__global__ void Updating_label(int* reduce_label, int* reduce_label_count, int* updating, int* labels_gpu,int GRAPHSIZE,int BLOCK_PER_VER)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= GRAPHSIZE)
        return;
    int cont = 1, label = labels_gpu[tid];
    int start = tid * BLOCK_PER_VER, end = start + BLOCK_PER_VER;
    for (int i = start; i < end; ++i)
    {
        if (reduce_label_count[i] > cont)
        {
            cont = reduce_label_count[i];
            label = reduce_label[i];
        }
        else if (reduce_label_count[i] == cont && reduce_label[i] < label)
        {
            label = reduce_label[i];
        }
    }
    if (label != labels_gpu[tid])
        *updating = 1;
    labels_gpu[tid] = label;
    return;
}

int Community_Detection(graph_structure& graph)
{
    make_csr(graph,GRAPHSIZE);

    int BLOCK_PER_VER=((GRAPHSIZE + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    int REDUCE_BLOCK_PER_GRID=(GRAPHSIZE * BLOCK_PER_VER + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    dim3 blockPerGrid((GRAPHSIZE + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, 1, 1);
    dim3 useBlock((GRAPHSIZE + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK * GRAPHSIZE, 1, 1);
    dim3 threadPerBlock(THREAD_PER_BLOCK, 1, 1);
    dim3 reduceBlock(REDUCE_BLOCK_PER_GRID, 1, 1);

    cudaMalloc(&row_ptr_gpu, (GRAPHSIZE + 1) * sizeof(int));
    cudaMalloc(&labels_gpu, GRAPHSIZE * sizeof(int));
    cudaMalloc(&neighbor_gpu, neighbor.size() * sizeof(int));
    cudaMalloc(&reduce_label, GRAPHSIZE * BLOCK_PER_VER * sizeof(int));
    cudaMalloc(&reduce_label_count, GRAPHSIZE * BLOCK_PER_VER * sizeof(int));
    cudaMemcpy(row_ptr_gpu, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_gpu, neighbor.data(), neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMallocManaged(&updating, sizeof(int));
    
    int it=0;
    float CUDAtime = 0;
    *updating = 1;
    init_label << <blockPerGrid, threadPerBlock >> > (labels_gpu,GRAPHSIZE);
    cudaDeviceSynchronize();
    cudaEvent_t GPUstart, GPUstop;
    cudaEventCreate(&GPUstart);
    cudaEventCreate(&GPUstop);
    cudaEventRecord(GPUstart, 0);
    while (*updating)
    {
        it++;
        *updating = 0;
        LPA << <useBlock, threadPerBlock, sizeof(int)* THREAD_PER_BLOCK >> > (row_ptr_gpu, labels_gpu, neighbor_gpu, reduce_label, reduce_label_count,GRAPHSIZE,BLOCK_PER_VER);
        cudaDeviceSynchronize();
        Updating_label << <reduceBlock, threadPerBlock >> > (reduce_label, reduce_label_count, updating, labels_gpu,GRAPHSIZE,BLOCK_PER_VER);
        cudaDeviceSynchronize();
        cout<<"it :"<<it<<endl;
    }

    cudaEventRecord(GPUstop, 0);
    cudaEventSynchronize(GPUstop);

    float timp = 0;
    cudaEventElapsedTime(&timp, GPUstart, GPUstop);
    CUDAtime += timp;
    cudaEventDestroy(GPUstart);
    cudaEventDestroy(GPUstop);

    cout << "CUDA time :" << CUDAtime << " ms" << endl;
}

int main(){
    std::string file_path;
    std::cout << "Please input the file path of the graph: ";
    std::cin >> file_path;
    graph_structure graph;
    graph.read_txt(file_path);
    Community_Detection(graph);
    return 0;
}