#include <GPU_Community_Detection.cuh>
using namespace std;

static int CD_GRAPHSIZE;
static int CD_ITERATION;
static vector<int> outs_ptr,ins_ptr, outs_neighbor,ins_neighbor;
// static vector<int> neighbor;
static int* outs_ptr_gpu,*ins_ptr_gpu;
static int* labels_gpu, * outs_neighbor_gpu,*ins_neighbor_gpu;
static int* reduce_label, * reduce_label_count;

// static int* updating;

template <typename T>
void make_csr(graph_structure<T> &graph, int& CD_GRAPHSIZE)
{
   
    CSR_graph<T> ARRAY_graph;
    ARRAY_graph=graph.toCSR();

    CD_GRAPHSIZE = ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;

    outs_ptr.resize(CD_GRAPHSIZE + 1);
    outs_ptr=ARRAY_graph.OUTs_Neighbor_start_pointers;
    ins_ptr.resize(CD_GRAPHSIZE + 1);
    ins_ptr=ARRY_graph.INs_Neighbor_start_pointers;

    outs_neighbor=ARRAY_graph.OUTs_Edges;
    ins_neighbor=ARRAY_graph.INs_Edges;

}


__global__ void init_label(int* labels_gpu,int CD_GRAPHSIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < CD_GRAPHSIZE)
    {
        labels_gpu[tid] = tid;
    }
}

__global__ void LPA(int* outs_ptr_gpu, int* labels_gpu, int* outs_neighbor_gpu, int* reduce_label, int* reduce_label_count,int CD_GRAPHSIZE,int BLOCK_PER_VER,int * ins_ptr_gpu,int* ins_neighbor_gpu)
{
    extern __shared__ int label_counts[];
    extern __shared__ int label[];
    int block_order=blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int ver = block_order / BLOCK_PER_VER;
    if(ver>=CD_GRAPHSIZE) return;
    int segment_order = block_order % BLOCK_PER_VER;
    int tid = (segment_order) * blockDim.x + threadIdx.x;
    
    int stid = threadIdx.x;
    if (tid == ver)
    {
        label_counts[stid] = 1;
    }
    else
    {
        label_counts[stid] = 0;
    }
    label[stid] = tid;

    __syncthreads();

    int outs_start = outs_ptr_gpu[ver], outs_end = outs_ptr_gpu[ver + 1];
    if (tid < outs_end - outs_start)
    {
        int neighbor_label = labels_gpu[outs_neighbor_gpu[start + tid]];
        if (neighbor_label >= segment_order * CD_THREAD_PER_BLOCK && neighbor_label < (segment_order + 1) * CD_THREAD_PER_BLOCK)
            atomicAdd(&label_counts[neighbor_label - segment_order * CD_THREAD_PER_BLOCK], 1);
    }
    int ins_start = ins_ptr_gpu[ver], ins_end = ins_ptr_gpu[ver + 1];
    if (tid < ins_end - ins_start)
    {
        int neighbor_label = labels_gpu[ins_neighbor_gpu[start + tid]];
        if (neighbor_label >= segment_order * CD_THREAD_PER_BLOCK && neighbor_label < (segment_order + 1) * CD_THREAD_PER_BLOCK)
            atomicAdd(&label_counts[neighbor_label - segment_order * CD_THREAD_PER_BLOCK], 1);
    }


    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (stid < s)
        {
            if (label_counts[stid] < label_counts[stid + s])
            {
                label_counts[stid] = label_counts[stid + s];
                label[stid] = label_counts[stid + s];
            }
            else if (label_counts[stid] == label_counts[stid + s] && label[stid] > label_counts[stid + s])
            {
                label[stid] = label_counts[stid + s];
            }
        }
        __syncthreads();
    }
    reduce_label_count[block_order] = label_counts[0];
    reduce_label[block_order] = label[0];
    return;
}

__global__ void Updating_label(int* reduce_label, int* reduce_label_count,  int* labels_gpu,int CD_GRAPHSIZE,int BLOCK_PER_VER)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= CD_GRAPHSIZE)
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

    labels_gpu[tid] = label;
    return;
}

int Community_Detection(graph_structure<double>& graph, float* elapsedTime)
{
    make_csr(graph,CD_GRAPHSIZE);
    CD_ITERATION=graph.cdlp_max_its;
    int BLOCK_PER_VER=((CD_GRAPHSIZE + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK);
    int REDUCE_BLOCK_PER_GRID=(CD_GRAPHSIZE + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK;

    dim3 blockPerGrid((CD_GRAPHSIZE + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1);
    dim3 useBlock(1e9/BLOCK_PER_VER*BLOCK_PER_VER, 1e4, 1e4);
    dim3 threadPerBlock(CD_THREAD_PER_BLOCK, 1, 1);
    dim3 reduceBlock(REDUCE_BLOCK_PER_GRID, 1, 1);

    cudaMalloc(&outs_ptr_gpu, (CD_GRAPHSIZE + 1) * sizeof(int));
    cudaMalloc(&ins_ptr_gpu, (CD_GRAPHSIZE + 1) * sizeof(int));
    cudaMalloc(&labels_gpu, CD_GRAPHSIZE * sizeof(int));
    cudaMalloc(&outs_neighbor_gpu, outs_neighbor.size() * sizeof(int));
    cudaMalloc(&ins_neighbor_gpu, ins_neighbor.size() * sizeof(int));
    cudaMalloc(&reduce_label, CD_GRAPHSIZE * BLOCK_PER_VER * sizeof(int));
    cudaMalloc(&reduce_label_count, CD_GRAPHSIZE * BLOCK_PER_VER * sizeof(int));
    cudaMemcpy(outs_ptr_gpu, outs_ptr.data(), outs_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ins_ptr_gpu, ins_ptr.data(), ins_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(outs_neighbor_gpu, outs_neighbor.data(), outs_neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ins_neighbor_gpu, ins_neighbor.data(), ins_neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMallocManaged(&updating, sizeof(int));
    
    int it=0;
    // *updating = 1;
    init_label << <blockPerGrid, threadPerBlock >> > (labels_gpu,CD_GRAPHSIZE);
    cudaDeviceSynchronize();
    cudaEvent_t GPUstart, GPUstop;
    cudaEventCreate(&GPUstart);
    cudaEventCreate(&GPUstop);
    cudaEventRecord(GPUstart, 0);
    while (it<CD_ITERATION)
    {
        it++;
        LPA << <useBlock, threadPerBlock, sizeof(int)* CD_THREAD_PER_BLOCK*2 >> > (outs_ptr_gpu, labels_gpu, outs_neighbor_gpu, reduce_label, reduce_label_count,CD_GRAPHSIZE,BLOCK_PER_VER,ins_ptr_gpu,ins_neighbor_gpu);
        cudaDeviceSynchronize();
        Updating_label << <reduceBlock, threadPerBlock >> > (reduce_label, reduce_label_count,  labels_gpu,CD_GRAPHSIZE,BLOCK_PER_VER);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(GPUstop, 0);
    cudaEventSynchronize(GPUstop);


    cudaEventElapsedTime(elapsedTime, GPUstart, GPUstop);

    cudaEventDestroy(GPUstart);
    cudaEventDestroy(GPUstop);

    cudaFree(outs_ptr_gpu);
    cudaFree(labels_gpu);
    cudaFree(outs_neighbor_gpu);
    cudaFree(reduce_label);
    cudaFree(reduce_label_count);

    return 0;
}
