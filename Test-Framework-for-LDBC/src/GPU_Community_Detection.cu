#include <GPU_Community_Detection.cuh>
using namespace std;

static int CD_GRAPHSIZE;
static int CD_ITERATION;
static int CD_SET_THREAD;
// static int CD_M;
static vector<int> outs_ptr, ins_ptr, outs_neighbor, ins_neighbor, in_out_ptr;
//The pointer indicates the starting and ending positions of the vertex's outgoing and incoming edges
static int *in_out_ptr_gpu;
static int *outs_ptr_gpu, *ins_ptr_gpu;
static int *outs_neighbor_gpu, *ins_neighbor_gpu;
static int *new_labels_gpu, *labels_gpu;
static int *global_space_for_label;

template <typename T>
void pre_set(graph_structure<T> &graph, int &CD_GRAPHSIZE)
{
    
    CSR_graph<T> ARRAY_graph = toCSR(graph);

    CD_ITERATION = graph.cdlp_max_its;
    CD_GRAPHSIZE = ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;
    // CD_SET_THREAD = 100;
    //Preprocessing range pointer
    outs_ptr.resize(CD_GRAPHSIZE + 1);
    outs_ptr = ARRAY_graph.OUTs_Neighbor_start_pointers;
    ins_ptr.resize(CD_GRAPHSIZE + 1);
    ins_ptr = ARRAY_graph.INs_Neighbor_start_pointers;
    in_out_ptr.resize(CD_GRAPHSIZE + 1);//for all Adjacency edge
    in_out_ptr[0] = 0;
    for (int i = 1; i <= CD_GRAPHSIZE; ++i)
    {
        in_out_ptr[i] = in_out_ptr[i - 1] + (ins_ptr[i] - ins_ptr[i - 1]) + (outs_ptr[i] - outs_ptr[i - 1])+1;
    }
    // for(int i=0;i<10000;++i){
    //     cout<<outs_ptr[i]<<"  "<<ins_ptr[i]<<"  "<<in_out_ptr[i]<<endl;
    // }
    outs_neighbor = ARRAY_graph.OUTs_Edges;//copy data
    ins_neighbor = ARRAY_graph.INs_Edges;

    // int sum = 0;
    // int max = 0;
    // for (int i = 0; i < CD_GRAPHSIZE; ++i)
    // {
    //     int cont = 0;
    //     cont += (ins_ptr[i + 1] - ins_ptr[i] + outs_ptr[i + 1] - outs_ptr[i]);
    //     if (cont > max)
    //     {
    //         max = cont;
    //     }
    //     sum += cont;
    // }
    // cout << "max degree : " << max << endl;
    // cout << "avg degree : " << sum / CD_GRAPHSIZE << endl;
    // int t = (size_t)(20LL * (1LL << 30)) / (max * 2 * 4);
    // cout << "use 20GB for max degree : " << t << endl;

    // CD_M = max;
    // CD_SET_THREAD = t > 10000 ? 10000 : t;
}

__global__ void init_label(int *labels_gpu,int *new_labels_gpu, int CD_GRAPHSIZE)
{
    //Initialize all labels at once with GPU.Initially, each vertex v is assigned a unique label which matches its identifier.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < CD_GRAPHSIZE)
    {
        labels_gpu[tid] = tid;
        new_labels_gpu[tid]=tid;
    }
}

__global__ void extract_labels(int *in_out_ptr_gpu, int *ins_ptr_gpu, int *outs_ptr_gpu, int *ins_neighbor_gpu, int *outs_neighbor_gpu, int *labels, int *labels_out, int CD_GRAPHSIZE)
{   //This function is used to obtain the labels of vertex neighbors
    int tid = blockIdx.x * blockDim.x + threadIdx.x;//Calculate which vertex to process
    if (tid >= CD_GRAPHSIZE)
        return;

    int start = in_out_ptr_gpu[tid];//start pointer of all edges
    int len_out = outs_ptr_gpu[tid + 1] - outs_ptr_gpu[tid];//length of out edges
    int len_in = ins_ptr_gpu[tid + 1] - ins_ptr_gpu[tid];//length of in edges
    // int end = in_out_ptr_gpu[tid + 1];
    for (int i = 0; i < len_in; ++i)
    {
        //get neighbor's location->get neighbor->get label
        labels_out[start + i] = labels[ins_neighbor_gpu[ins_ptr_gpu[tid] + i]];
    }
    
    for (int i = 0; i < len_out; ++i)
    {
        labels_out[start + len_in + i] = labels[outs_neighbor_gpu[outs_ptr_gpu[tid] + i]];
    }
    labels_out[start + len_in+len_out ] = labels[tid];//vertex it self 

    return;
}

__global__ void parallel_sort_labels(int *in_out_ptr_gpu, int *labels_out, int CD_GRAPHSIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 10)
        return;

    int start = in_out_ptr_gpu[tid];
    int end = in_out_ptr_gpu[tid + 1];
    thrust::sort(thrust::device, labels_out + start, labels_out + end);
    //thrust::sort is a cuda function
    __syncthreads();
    /* __Syncthreads() is a crucial synchronization primitive in CUDA programming. 
    Its function is to ensure that all threads in a certain thread block have 
    completed all their previous instructions before executing this function. 
    Once all threads reach this synchronization point, 
    they can continue executing instructions after __ syncthreads. */
}

inline void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

// QuickSort function for raw pointers
void quickSort(int* arr, int low, int high) {
    //Using Fast and Slow Pointer Method to Complete QuickSorti
    if (low < high) {
        int pivot = arr[high]; // choosing the last element as pivot
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        int pi = i + 1;

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
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

__global__ void LPA(int *global_space_for_label, int *in_out_ptr_gpu, int *labels_gpu, int *new_labels_gpu, int CD_GRAPHSIZE)
{   //Use GPU to propagate all labels at the same time.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= CD_GRAPHSIZE)
        return;
    int start = in_out_ptr_gpu[tid], end = in_out_ptr_gpu[tid + 1];
    int current_label = -1;
    int current_count = 0;
    int max_label = current_label;
    int max_count = current_count;
    for (int i = start; i < end; ++i)
    {
        /* After the previous sorting, the labels in the array are now ordered, 
        so the number of occurrences is calculated in order for each label */
        if (global_space_for_label[i] == current_label)
        {   //If it is still the label from the previous calculation, continue to increase it
            current_count++;
        }
        else
        {
            //If the label changes,update the most frequently label and start accumulating again from 1
            if (current_count > max_count)
            {   
                max_count = current_count;
                max_label = current_label;
            }
            else if (current_count == max_count && current_label < max_label)
            {
                max_label = current_label;//In case there are multiple labels with the maximum frequency, the smallest label is chosen. 
            }
            current_label = global_space_for_label[i];
            current_count = 1;
        }
    }
    if (current_count > max_count)
    {
        max_count = current_count;
        max_label = current_label;
    }
    else if (current_count == max_count && current_label < max_label)
    {
        max_label = current_label;  //In case there are multiple labels with the maximum frequency, the smallest label is chosen. 
    }
    new_labels_gpu[tid] = max_label;
}

int gpu_Community_Detection(graph_structure<double> &graph, float *elapsedTime, vector<int> &ans)
{
    pre_set(graph, CD_GRAPHSIZE);

    dim3 init_label_block((CD_GRAPHSIZE + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1);
    dim3 init_label_thread(CD_THREAD_PER_BLOCK, 1, 1);
    // dim3 LPA_block((CD_SET_THREAD + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1);
    // dim3 LPA_thread(CD_THREAD_PER_BLOCK, 1, 1);

    // cout << 1 << endl;
    cudaMalloc(&outs_ptr_gpu, (CD_GRAPHSIZE + 1) * sizeof(int));
    cudaMalloc(&ins_ptr_gpu, (CD_GRAPHSIZE + 1) * sizeof(int));
    cudaMallocManaged(&labels_gpu, CD_GRAPHSIZE * sizeof(int));
    cudaMallocManaged(&new_labels_gpu, CD_GRAPHSIZE * sizeof(int));
    cudaMalloc(&outs_neighbor_gpu, outs_neighbor.size() * sizeof(int));
    cudaMalloc(&ins_neighbor_gpu, ins_neighbor.size() * sizeof(int));
    cudaMallocManaged(&global_space_for_label, (outs_neighbor.size() + ins_neighbor.size() + CD_GRAPHSIZE) * sizeof(int));
    cudaMalloc(&in_out_ptr_gpu, (CD_GRAPHSIZE + 1) * sizeof(int));
    //allocate cuda memory and copy data from host to device
    cudaMemcpy(outs_ptr_gpu, outs_ptr.data(), outs_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ins_ptr_gpu, ins_ptr.data(), ins_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(outs_neighbor_gpu, outs_neighbor.data(), outs_neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ins_neighbor_gpu, ins_neighbor.data(), ins_neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_out_ptr_gpu, in_out_ptr.data(), in_out_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);

    // checkDeviceProperties();
    // get_size();
    // cout << 2 << endl;
    cudaError_t err;
    init_label<<<init_label_block, init_label_thread>>>(labels_gpu,new_labels_gpu, CD_GRAPHSIZE);
    err = cudaDeviceSynchronize();
    checkCudaError(err, "cudaDeviceSynchronize after init_label");
    // cout << 3 << endl;

    cudaEvent_t GPUstart, GPUstop;
    cudaEventCreate(&GPUstart);
    cudaEventCreate(&GPUstop);
    cudaEventRecord(GPUstart, 0);

    int it = 0;
    // cout << 4 << endl;

    // CD_ITERATION=3;
    while (it < CD_ITERATION)
    {
        //The difference between if and else lies in the use of different old and new pointers
        if (it % 2 == 0)
        {
            //Calculate the neighbor label array for each vertex
            extract_labels<<<init_label_block, init_label_thread>>>(in_out_ptr_gpu, ins_ptr_gpu, outs_ptr_gpu, ins_neighbor_gpu, outs_neighbor_gpu, labels_gpu, global_space_for_label, CD_GRAPHSIZE);
            err = cudaDeviceSynchronize();
            checkCudaError(err, "cudaDeviceSynchronize after extract_labels");
            // for(int i=0;i<10;++i){
            //     //cout<<ins_ptr[i+1]<<"  "<<ins_ptr[i]<<"  "<<outs_ptr[i+1]<<"  "<<outs_ptr[i]<<"  "<<ins_ptr[i+1]-ins_ptr[i]<<"  "<<outs_ptr[i+1]-outs_ptr[i]<<"  "<<in_out_ptr[i]<<"  "<<in_out_ptr[i+1]<<endl;
            //     for(int j=in_out_ptr[i];j<in_out_ptr[i+1];j++){
            //         cout<<global_space_for_label[j]<<"   ";
            //     }
            //     cout<<endl<<"--------------------------------------------------------------"<<endl;
            // }
            // cout<<"************************************************"<<endl;
            // parallel_sort_labels<<<init_label_block, init_label_thread>>>(in_out_ptr_gpu, global_space_for_label, CD_GRAPHSIZE);
            // err = cudaDeviceSynchronize();
            // checkCudaError(err, "cudaDeviceSynchronize after parallel_sort_labels");
            for(int i=0;i<CD_GRAPHSIZE;++i){
                quickSort(global_space_for_label,in_out_ptr[i],  in_out_ptr[i+1]-1);//Sort the array of neighbor labels for each vertex
            }
            
            // for(int i=0;i<10;++i){
            //     //cout<<ins_ptr[i+1]<<"  "<<ins_ptr[i]<<"  "<<outs_ptr[i+1]<<"  "<<outs_ptr[i]<<"  "<<ins_ptr[i+1]-ins_ptr[i]<<"  "<<outs_ptr[i+1]-outs_ptr[i]<<"  "<<in_out_ptr[i]<<"  "<<in_out_ptr[i+1]<<endl;
            //     for(int j=in_out_ptr[i];j<in_out_ptr[i+1];j++){
            //         cout<<global_space_for_label[j]<<"   ";
            //     }
            //     cout<<endl<<"--------------------------------------------------------------"<<endl;
            // }
            //Calculate vertex labels for this round
            LPA<<<init_label_block, init_label_thread>>>(global_space_for_label, in_out_ptr_gpu, labels_gpu, new_labels_gpu, CD_GRAPHSIZE);
            err = cudaDeviceSynchronize();
            checkCudaError(err, "cudaDeviceSynchronize after LPA");
        }
        else
        {
            extract_labels<<<init_label_block, init_label_thread>>>(in_out_ptr_gpu, ins_ptr_gpu, outs_ptr_gpu, ins_neighbor_gpu, outs_neighbor_gpu, new_labels_gpu, global_space_for_label, CD_GRAPHSIZE);
            err = cudaDeviceSynchronize();
            checkCudaError(err, "cudaDeviceSynchronize after extract_labels");
            // cout<<new_labels_gpu[1852854]<<endl;
            // for(int i=1000;i<1010;++i){
            //     cout<<ins_ptr[i+1]<<"  "<<ins_ptr[i]<<"  "<<outs_ptr[i+1]<<"  "<<outs_ptr[i]<<"  "<<ins_ptr[i+1]-ins_ptr[i]<<"  "<<outs_ptr[i+1]-outs_ptr[i]<<"  "<<in_out_ptr[i]<<"  "<<in_out_ptr[i+1]<<endl;
            //     for(int j=in_out_ptr[i];j<in_out_ptr[i+1];j++){
            //         cout<<global_space_for_label[j]<<"   ";
            //     }
            //     cout<<endl<<"--------------------------------------------------------------"<<endl;
            // }
            // cout<<"************************************************"<<endl;
            // parallel_sort_labels<<<init_label_block, init_label_thread>>>(in_out_ptr_gpu, global_space_for_label, CD_GRAPHSIZE);
            // err = cudaDeviceSynchronize();
            // checkCudaError(err, "cudaDeviceSynchronize after parallel_sort_labels");
            for(int i=0;i<CD_GRAPHSIZE;++i){
                quickSort(global_space_for_label,in_out_ptr[i],  in_out_ptr[i+1]-1);
            }
            // for(int i=1000;i<1010;++i){
            //     cout<<ins_ptr[i+1]<<"  "<<ins_ptr[i]<<"  "<<outs_ptr[i+1]<<"  "<<outs_ptr[i]<<"  "<<ins_ptr[i+1]-ins_ptr[i]<<"  "<<outs_ptr[i+1]-outs_ptr[i]<<"  "<<in_out_ptr[i]<<"  "<<in_out_ptr[i+1]<<endl;
            //     for(int j=in_out_ptr[i];j<in_out_ptr[i+1];j++){
            //         cout<<global_space_for_label[j]<<"   ";
            //     }
            //     cout<<endl<<"--------------------------------------------------------------"<<endl;
            // }
            LPA<<<init_label_block, init_label_thread>>>(global_space_for_label, in_out_ptr_gpu, new_labels_gpu, labels_gpu, CD_GRAPHSIZE);
            err = cudaDeviceSynchronize();
            checkCudaError(err, "cudaDeviceSynchronize after LPA");
        }

        it++;
    }

    cudaEventRecord(GPUstop, 0);
    cudaEventSynchronize(GPUstop);

    cudaEventElapsedTime(elapsedTime, GPUstart, GPUstop);
    cudaEventDestroy(GPUstart);
    cudaEventDestroy(GPUstop);

    ans.resize(CD_GRAPHSIZE);
    if(CD_ITERATION%2==0){
        cudaMemcpy(ans.data(), labels_gpu, CD_GRAPHSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    }else{
        cudaMemcpy(ans.data(), new_labels_gpu, CD_GRAPHSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    cudaFree(outs_ptr_gpu);
    cudaFree(ins_ptr_gpu);
    cudaFree(labels_gpu);
    cudaFree(outs_neighbor_gpu);
    cudaFree(ins_neighbor_gpu);
    cudaFree(new_labels_gpu);

    return 0;
}
