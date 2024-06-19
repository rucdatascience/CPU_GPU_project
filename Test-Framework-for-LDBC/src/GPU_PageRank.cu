#include <GPU_PageRank.cuh>
static int ITERATION;
static int N, E_in,E_out;

static vector<int> sink_vertexs; // sink vertexs
static int *sink_vertex_gpu;
static double *pr, *npr, *outs;
static int out_zero_size;
static double *sink_sum;
static double damp, teleport;
static int *in_pointer, *out_pointer, *in_edge, *out_edge;
void GPU_PR(graph_structure<double> &graph, float *elapsedTime, vector<double> &result)
{
    ITERATION = graph.pr_its;
    CSR_graph<double> ARRAY_graph = graph.toCSR();
    damp = graph.pr_damping;
    N = graph.size();
    E_in = ARRAY_graph.INs_Edges.size();
    E_out = ARRAY_graph.OUTs_Edges.size();
    teleport = (1 - damp) / N;

    cudaMallocManaged(&in_pointer, (N + 1) * sizeof(int));
    cudaMallocManaged(&out_pointer, (N + 1) * sizeof(int));
    cudaMallocManaged(&outs, N * sizeof(double));
    cudaMallocManaged(&sink_sum, sizeof(double));
    cudaMallocManaged(&npr, N * sizeof(double));
    cudaMallocManaged(&pr, N * sizeof(double));
    cudaMallocManaged(&in_edge, E_in * sizeof(int));
    cudaMallocManaged(&out_edge, E_out * sizeof(int));

    cudaMemcpy(in_pointer, ARRAY_graph.INs_Neighbor_start_pointers.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_pointer, ARRAY_graph.OUTs_Neighbor_start_pointers.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_edge, ARRAY_graph.INs_Edges.data(), E_in* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_edge, ARRAY_graph.OUTs_Edges.data(), E_out * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < N; i++)
    {
        if (ARRAY_graph.OUTs_Neighbor_start_pointers[i] == ARRAY_graph.OUTs_Neighbor_start_pointers[i + 1])
        {
            // This means that the vertex has no edges
            sink_vertexs.push_back(i);
        }
    }
    out_zero_size = sink_vertexs.size();
    cudaMallocManaged(&sink_vertex_gpu, sink_vertexs.size() * sizeof(int));
    cudaMemcpy(sink_vertex_gpu, sink_vertexs.data(), sink_vertexs.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockPerGrid((N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, 1, 1);
    dim3 threadPerGrid(THREAD_PER_BLOCK, 1, 1);

    int iteration = 0;
    cudaEvent_t GPUstart, GPUstop; // record GPU_TIME
    cudaEventCreate(&GPUstart);
    cudaEventCreate(&GPUstop);
    cudaEventRecord(GPUstart, 0);
    initialization<<<blockPerGrid, threadPerGrid>>>(pr, outs, out_pointer, N);
    while (iteration < ITERATION)
    {
        *sink_sum = 0;
        calculate_sink<<<blockPerGrid, threadPerGrid, THREAD_PER_BLOCK * sizeof(double)>>>(pr, sink_vertex_gpu, out_zero_size, sink_sum);
        cudaDeviceSynchronize();
        *sink_sum = (*sink_sum) * damp / N;
        Antecedent_division<<<blockPerGrid, threadPerGrid>>>(pr, outs, N);
        cudaDeviceSynchronize();
        importance<<<blockPerGrid, threadPerGrid>>>(npr, pr, teleport, *sink_sum, damp, in_edge, in_pointer, N);
        cudaDeviceSynchronize();

        std::swap(pr, npr);
        iteration++;
    }
    // get gpu PR algorithm result
    double *gpu_res = new double[N];
    cudaMemcpy(gpu_res, pr, N * sizeof(double), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < GRAPHSIZE; ++i){
    //     cout<<"the gpu_res is:"<<gpu_res[i]<<endl;
    // }
    std::copy(gpu_res, gpu_res + N, std::back_inserter(result));

    cudaEventRecord(GPUstop, 0);
    cudaEventSynchronize(GPUstop);

    float CUDAtime = 0;
    cudaEventElapsedTime(&CUDAtime, GPUstart, GPUstop);
    *elapsedTime += CUDAtime;
    cudaEventDestroy(GPUstart);
    cudaEventDestroy(GPUstop);
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

__global__ void Antecedent_division(double *pr, double *outs, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < N)
    {
        pr[tid] *= outs[tid];
    }
}

__global__ void importance(double *npr, double *pr, double tele, double red, double damp, int *in_edge, int *in_pointer, int GRAPHSIZE)
{                                                    // importance
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex
    if (tid >= 0 && tid < GRAPHSIZE)
    {
        int rbegin = in_pointer[tid];
        int rend = in_pointer[tid + 1];
        // begin and end of in edges
        double acc = 0; // sum of u belongs to Nin(v)
        for (int c = rbegin; c < rend; c++)
        { // val_col[c] is neighbor,rank get PR(u) row_value is denominator i.e. Nout
            acc += pr[in_edge[c]];
        }
        // printf("tid : %d  acc : %f\n", tid, acc);
        npr[tid] = acc * damp + red + tele; // scaling is damping factor
    }
    return;
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