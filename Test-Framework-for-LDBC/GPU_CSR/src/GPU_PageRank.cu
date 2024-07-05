#include <GPU_PageRank.cuh>
static int ITERATION;
static int N;

static vector<int> sink_vertexs; // sink vertexs
static int *sink_vertex_gpu;
static double *pr, *npr, *outs;
static int out_zero_size;
static double *sink_sum;
static double damp, teleport;
dim3 blockPerGrid,threadPerGrid;
// void GPU_PR(graph_structure<double> &graph, float *elapsedTime, vector<double> &result,int *in_pointer, int *out_pointer,int *in_edge,int *out_edge)
void GPU_PR(LDBC<double> &graph, float *elapsedTime, vector<double> &result,int *in_pointer, int *out_pointer,int *in_edge,int *out_edge)
{
    ITERATION = graph.pr_its;
    damp = graph.pr_damping;
    N = graph.size();
    teleport = (1 - damp) / N;


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
    // cudaEvent_t GPUstart, GPUstop; // record GPU_TIME
    // cudaEventCreate(&GPUstart);
    // cudaEventCreate(&GPUstop);
    // cudaEventRecord(GPUstart, 0);
    initialization<<<blockPerGrid, threadPerGrid>>>(pr, outs, out_pointer, N);
    while (iteration < ITERATION)
    {
        *sink_sum = 0;
        calculate_sink<<<blockPerGrid, threadPerGrid, THREAD_PER_BLOCK * sizeof(double)>>>(pr, sink_vertex_gpu, out_zero_size, sink_sum);
        cudaDeviceSynchronize();
        *sink_sum = (*sink_sum) * damp / N;
        Antecedent_division<<<blockPerGrid, threadPerGrid>>>(pr, npr,outs,teleport+(*sink_sum), N);
        cudaDeviceSynchronize();
        importance<<<blockPerGrid, threadPerGrid>>>(npr, pr,  damp, in_edge, in_pointer, N);
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

    // cudaEventRecord(GPUstop, 0);
    // cudaEventSynchronize(GPUstop);

    // float CUDAtime = 0;
    // cudaEventElapsedTime(&CUDAtime, GPUstart, GPUstop);
    // *elapsedTime += CUDAtime;
    // cudaEventDestroy(GPUstart);
    // cudaEventDestroy(GPUstop);


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
        double acc=0; // sum of u belongs to Nin(v)
        //double *acc_point = &acc;
         for (int c = in_pointer[tid]; c < in_pointer[tid+1]; c++)
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

std::map<long long int, double> getGPUPR(LDBC<double> & graph, CSR_graph<double> & csr_graph){
    vector<double> gpuPrVec(graph.size());
    GPU_PR(graph, 0, gpuPrVec,csr_graph.in_pointer,csr_graph.out_pointer,csr_graph.in_edge,csr_graph.out_edge);
    std::map<long long int, double> strId2value;

    std::vector<long long int> converted_numbers;

    for (const auto& str : graph.vertex_id_to_str) {
        long long int num = std::stoll(str);
        converted_numbers.push_back(num);
    }

    std::sort(converted_numbers.begin(), converted_numbers.end());

	for( int i = 0; i < gpuPrVec.size(); ++i){
		strId2value.emplace(converted_numbers[i], gpuPrVec[i]);
    }

	// std::string path = "../data/cpu_pr_75.txt";
	// storeResult(strId2value, path);//ldbc file

    return strId2value;
}

std::vector<std::string> GPU_PR_v2(LDBC<double> & graph, CSR_graph<double> &csr_graph){
    vector<double> gpuPrVec(graph.size());
    GPU_PR(graph, 0, gpuPrVec,csr_graph.in_pointer,csr_graph.out_pointer,csr_graph.in_edge,csr_graph.out_edge);

    std::vector<std::string> resultVec(graph.size());

    for(auto & it : gpuPrVec){
		resultVec.push_back(std::to_string(it));
	}

	return resultVec;
}

void GPU_PR_v3(LDBC<double> &graph, float *elapsedTime, std::vector<std::string> &result,int *in_pointer, int *out_pointer,int *in_edge,int *out_edge){
    vector<double> gpuPrVec(graph.size());

    // std::cout<<"PR V3 before size ="<<gpuPrVec.size()<<std::endl;

    GPU_PR(graph, elapsedTime, gpuPrVec, in_pointer, out_pointer, in_edge, out_edge);
    
    // std::cout<<"PR V3 size ="<<gpuPrVec.size()<<std::endl;
    
    for(int i = 0; i < graph.size(); ++i){
		result.push_back(std::to_string(gpuPrVec[i]));
    }

}
