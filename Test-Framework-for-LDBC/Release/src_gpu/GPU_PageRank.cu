#include <GPU_PageRank.cuh>
static int N;

static vector<int> sink_vertexs; // sink vertexs
static int *sink_vertex_gpu;
static double *pr, *npr, *outs;
static int out_zero_size;
static double *sink_sum;
static double teleport;
dim3 blockPerGrid,threadPerGrid;

void GPU_PR(graph_structure<double> &graph, CSR_graph<double>& csr_graph, vector<double>& result, int iterations, double damping)
{
    N = graph.V;
    teleport = (1 - damping) / N;

    int* in_pointer = csr_graph.in_pointer;
    int* out_pointer = csr_graph.out_pointer;
    int* in_edge = csr_graph.in_edge;
    int* out_edge = csr_graph.out_edge;

    cudaMallocManaged(&outs, N * sizeof(double));
    cudaMallocManaged(&sink_sum, sizeof(double));
    cudaMallocManaged(&npr, N * sizeof(double));
    cudaMallocManaged(&pr, N * sizeof(double));

    for (int i = 0; i < N; i++)
    {
        if (graph.OUTs[i].size()==0)
        {
            // This means that the vertex has no edges
            sink_vertexs.push_back(i);
        }
    }
    out_zero_size = sink_vertexs.size();
    cudaMallocManaged(&sink_vertex_gpu, sink_vertexs.size() * sizeof(int));
    cudaMemcpy(sink_vertex_gpu, sink_vertexs.data(), sink_vertexs.size() * sizeof(int), cudaMemcpyHostToDevice);
    blockPerGrid.x = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    threadPerGrid.x = THREAD_PER_BLOCK;

    int iteration = 0;

    initialization<<<blockPerGrid, threadPerGrid>>>(pr, outs, out_pointer, N);
    cudaDeviceSynchronize();
    while (iteration < iterations)
    {
        *sink_sum = 0;
        calculate_sink<<<blockPerGrid, threadPerGrid, THREAD_PER_BLOCK * sizeof(double)>>>(pr, sink_vertex_gpu, out_zero_size, sink_sum);
        cudaDeviceSynchronize();
        *sink_sum = (*sink_sum) * damping / N;
        Antecedent_division<<<blockPerGrid, threadPerGrid>>>(pr, npr, outs, teleport + (*sink_sum), N);
        cudaDeviceSynchronize();
        importance<<<blockPerGrid, threadPerGrid>>>(npr, pr, damping, in_edge, in_pointer, N);
        cudaDeviceSynchronize();

        std::swap(pr, npr);
        iteration++;
    }
    // get gpu PR algorithm result
    //double *gpu_res = new double[N];
    //cudaMemcpy(gpu_res, pr, N * sizeof(double), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < GRAPHSIZE; ++i){
    //     cout<<"the gpu_res is:"<<gpu_res[i]<<endl;
    // }
    //std::copy(gpu_res, gpu_res + N, std::back_inserter(result));

    result.resize(N);
    cudaMemcpy(result.data(), pr, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(pr);
    cudaFree(npr);
    cudaFree(outs);
    cudaFree(sink_vertex_gpu);
    cudaFree(sink_sum);
}

__global__ void initialization(double *pr, double *outs, int *out_pointer, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < N)
    {
        pr[tid] = 1 / N;
        if (out_pointer[tid + 1] - out_pointer[tid])
        {
            outs[tid] = 1 / (out_pointer[tid + 1] - out_pointer[tid]);
        }
        else
        {
            outs[tid] = 0;
        }
    }
}

__global__ void Antecedent_division(double *pr,double *npr, double *outs,double redi_tele, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < N)
    {
        pr[tid] *= outs[tid];
        npr[tid] = redi_tele;
    }
}

__global__ void importance(double *npr, double *pr,  double damp, int *in_edge, int *in_pointer, int GRAPHSIZE)
{                                                    // importance
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    
    if (tid >= 0 && tid < GRAPHSIZE)
    {

        // begin and end of in edges
        double acc = 0; // sum of u belongs to Nin(v)
        //double *acc_point = &acc;
         for (int c = in_pointer[tid]; c < in_pointer[tid + 1]; c++)
        { // val_col[c] is neighbor,rank get PR(u) row_value is denominator i.e. Nout
            acc += pr[in_edge[c]];
        } 
        // printf("tid : %d  acc : %f\n", tid, acc);
        npr[tid] = acc * damp; // scaling is damping factor
    }
    return;
}

__global__ void calculate_acc(double *pr,int *in_edge, int begin,int end,double *acc){
    extern __shared__ double temp[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stid = threadIdx.x;

    if (tid < end)
    {
        temp[stid] = pr[in_edge[tid+begin]];
    }
    else
    {
        temp[stid] = 0;
    }
    __syncthreads(); // wait unitl finish Loading data into shared memory

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (stid < i)
        {
            temp[stid] += temp[stid + i];
        }
        __syncthreads(); // Synchronize again to ensure that each step of the reduction operation is completed
    }
    if (stid == 0)
    {
        _atomicAdd(acc, temp[0]); // Write the result of each thread block into the output array
    }
}

__global__ void calculate_sink(double *pr, int *N_out_zero_gpu, int out_zero_size, double *sink_sum)
{
    // A reduction pattern was used to sum up
    extern __shared__ double sink[]; // Declare shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stid = threadIdx.x;

    if (tid < out_zero_size)
    {
        sink[stid] = pr[N_out_zero_gpu[tid]]; // get PR(w)
    }
    else
    {
        sink[stid] = 0;
    }
    __syncthreads(); // wait unitl finish Loading data into shared memory

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (stid < i)
        {
            sink[stid] += sink[stid + i];
        }
        __syncthreads(); // Synchronize again to ensure that each step of the reduction operation is completed
    }
    if (stid == 0)
    {
        _atomicAdd(sink_sum, sink[0]); // Write the result of each thread block into the output array
    }
}

__device__ double _atomicAdd(double *address, double val)
{
    /* Implementing atomic operations,
    that is, ensuring that adding operations to a specific
     memory location in a multi-threaded environment are thread safe. */
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

std::vector<std::pair<std::string, double>> Cuda_PR(graph_structure<double> &graph, CSR_graph<double> &csr_graph, int iterations, double damping){
    std::vector<double> result;
    GPU_PR(graph, csr_graph, result, iterations, damping);

    return graph.res_trans_id_val(result);
}