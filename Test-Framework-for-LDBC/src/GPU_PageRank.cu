#include <GPU_PageRank.cuh>

static int ITERATION;
static int ALPHA;
static int GRAPHSIZE;
static int *graphSize;
static int *row_point, *val_col;
static int *row_size;
static double *row_value;
static vector<int> N_out_zero;
static int *N_out_zero_gpu;
static int * row_out_ptr;
static vector<double> row_value_vec;
static vector<int> val_col_vec;
static int *verticeOrder;
static int *smallOffset, *normalOffset;
static double *Rank, *diff_array, *reduce_array;
static double *newRank, *F, *temp;
static int out_zero_size;
static double *sink_sum;
void PageRank(graph_structure<double> &graph, float *elapsedTime, vector<double> & result)
{
    CSR_graph<double> ARRAY_graph = graph.toCSR();
    GRAPHSIZE = ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;
    cudaMallocManaged(&graphSize, sizeof(int));
    cudaMallocManaged(&smallOffset, sizeof(int));
    cudaMallocManaged(&normalOffset, sizeof(int));
    cudaMallocManaged(&temp, sizeof(double *));
    cudaMallocManaged(&row_size, GRAPHSIZE * sizeof(int));
    cudaMallocManaged(&row_point, (GRAPHSIZE+1) * sizeof(int));
    cudaMallocManaged(&row_out_ptr, (GRAPHSIZE+1) * sizeof(int));
    cudaMallocManaged(&sink_sum, sizeof(double));
    cudaMallocManaged(&newRank, GRAPHSIZE * sizeof(double));
    cudaMallocManaged(&F, GRAPHSIZE * sizeof(double));
    cudaMallocManaged(&diff_array, GRAPHSIZE * sizeof(double));
    cudaMallocManaged(&reduce_array, GRAPHSIZE * sizeof(double));
    cudaMallocManaged(&Rank, GRAPHSIZE * sizeof(double));
    cudaMallocManaged(&verticeOrder, GRAPHSIZE * sizeof(int));
    cudaMemcpy(row_point, ARRAY_graph.INs_Neighbor_start_pointers.data(), (GRAPHSIZE+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_out_ptr, ARRAY_graph.OUTs_Neighbor_start_pointers.data(), (GRAPHSIZE+1) * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < GRAPHSIZE; i++)
    {
        for (auto it : graph.INs[i])
        {
            row_value_vec.push_back(1.0 / (graph.OUTs[it.first].size()));
            val_col_vec.push_back(it.first);
        }
        if(row_out_ptr[i]==row_out_ptr[i+1]){
            N_out_zero.push_back(i);
        }
    }
    cudaMallocManaged(&N_out_zero_gpu, N_out_zero.size() * sizeof(int));
    cudaMemcpy(N_out_zero_gpu, N_out_zero.data(),  N_out_zero.size() * sizeof(int), cudaMemcpyHostToDevice);
    out_zero_size=N_out_zero.size();
    ALPHA = graph.pr_damping;
    ITERATION = graph.pr_its;
    cudaMallocManaged(&row_value, row_value_vec.size() * sizeof(double));
    std::copy(row_value_vec.begin(), row_value_vec.end(), row_value);
    cudaMallocManaged(&val_col, val_col_vec.size() * sizeof(int));
    std::copy(val_col_vec.begin(), val_col_vec.end(), val_col);
    dim3 blockPerGrid((GRAPHSIZE + THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, 1, 1);
    dim3 threadPerGrid(THREAD_PER_BLOCK, 1, 1);

    for (int i = 0; i < GRAPHSIZE; i++)
    {
        Rank[i] = 1.0 / GRAPHSIZE;
    }
    int iteration = 0;
    double d = ALPHA, d_ops = (1 - ALPHA) / GRAPHSIZE;
    cudaEvent_t GPUstart, GPUstop;
    cudaEventCreate(&GPUstart);
    cudaEventCreate(&GPUstop);
    cudaEventRecord(GPUstart, 0);

    while (iteration < ITERATION)
    {
        *sink_sum=0;
        calculate_sink<<<blockPerGrid, threadPerGrid,THREAD_PER_BLOCK*sizeof(double)>>>(Rank, N_out_zero_gpu,out_zero_size,sink_sum);
        cudaDeviceSynchronize();
        
        tinySolve<<<blockPerGrid, threadPerGrid>>>(F, Rank, d, row_point, row_size, row_value, val_col, GRAPHSIZE);
        cudaDeviceSynchronize();
        
        add_scaling<<<blockPerGrid, threadPerGrid>>>(newRank, F, (ALPHA/GRAPHSIZE)*(*sink_sum)+d_ops, GRAPHSIZE);
        cudaDeviceSynchronize();

        temp = newRank;
        newRank = Rank;
        Rank = temp;
        iteration++;
    }
    //get gpu PR algorithm result
    cudaMemcpy(result.data(), Rank, GRAPHSIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(GPUstop, 0);
    cudaEventSynchronize(GPUstop);

    float CUDAtime = 0;
    cudaEventElapsedTime(&CUDAtime, GPUstart, GPUstop);
    *elapsedTime += CUDAtime;
    cudaEventDestroy(GPUstart);
    cudaEventDestroy(GPUstop);
}


bool cmp(const std::vector<pair<int, int>> &a, const std::vector<pair<int, int>> &b)
{
    return a.size() > b.size();
}

__global__ void add_scaling(double *newRank, double *oldRank, double scaling, int GRAPHSIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < GRAPHSIZE)
    {
        newRank[tid] = oldRank[tid] + scaling;
    }
    return;
}

__global__ void tinySolve(double *newRank, double *rank, double scaling, int *row_point, int *row_size, double *row_value, int *val_col, int GRAPHSIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < GRAPHSIZE)
    {
        int rbegin = row_point[tid];
        int rend = row_point[tid + 1];

        double acc = 0;
        for (int c = rbegin; c < rend; c++)
        {
            acc += row_value[c] * (rank[val_col[c]]);
        }
        // printf("tid : %d  acc : %f\n", tid, acc);
        newRank[tid] = acc * scaling;
    }
    return;
}
__device__ double _atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void calculate_sink(double* rank, int* N_out_zero_gpu, int out_zero_size, double* sink_sum) {
    extern __shared__ double sink[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stid = threadIdx.x;

    if (tid < out_zero_size) {
        sink[stid] = rank[N_out_zero_gpu[tid]];
    } else {
        sink[stid] = 0;
    }
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (stid < i) {
            sink[stid] += sink[stid + i];
        }
        __syncthreads();
    }

    if (stid == 0) {
        _atomicAdd(sink_sum, sink[0]);
    }
}


// __global__ void vec_diff(double *diff, double *newRank, double *oldRank)
// {

//     __shared__ double s_newRank[512];
//     __shared__ double s_oldRank[512];

//     int idx = threadIdx.x + blockIdx.x * blockDim.x;

//    
//     if (idx < GRAPHSIZE)
//     {

//         s_newRank[threadIdx.x] = newRank[idx];
//         s_oldRank[threadIdx.x] = oldRank[idx];

//         __syncthreads();

//         diff[idx] = abs(s_newRank[threadIdx.x] - s_oldRank[threadIdx.x]);
//     }
// }

// __global__ void reduce_kernel(double *input, double *output)
// {
//     extern __shared__ double sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < GRAPHSIZE)
//         sdata[tid] = input[i];
//     else
//         sdata[tid] = 0.0;
//     __syncthreads();

//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
//     {
//         if (tid < s)
//         {
//             // sdata[tid] += sdata[tid + s];
//             sdata[tid] = sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
//         }
//         __syncthreads();
//     }

//     if (tid == 0)
//         output[blockIdx.x] = sdata[0];
// }

// int main(){
//     std::string file_path;
//     std::cout << "Please input the file path of the graph: ";
//     std::cin >> file_path;
//     graph_structure<double> graph;
//     graph.read_txt(file_path);
//     PageRank(graph);
//     return 0;
// }