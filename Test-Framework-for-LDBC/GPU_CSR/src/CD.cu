#include <GPU_Community_Detection.cuh>
#include <time.h>
#include <omp.h>
#include<cub/cub.cuh>
#include<set>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
using namespace std;

static int CD_GRAPHSIZE;
static int CD_ITERATION;
static int CD_SET_THREAD;
// static int CD_M;
static vector<int> outs_ptr, ins_ptr, outs_neighbor, ins_neighbor, in_out_ptr;

static int *in_out_ptr_gpu;
static int *outs_ptr_gpu, *ins_ptr_gpu;
static int *outs_neighbor_gpu, *ins_neighbor_gpu;
static int *new_labels_gpu, *labels_gpu;
static int *global_space_for_label;
// static size_t temp_storage_bytes = 0;
// static int *d_temp_storage = nullptr;
template <typename T>
void pre_set(graph_structure<T> &graph, int &CD_GRAPHSIZE)
{

    CSR_graph<T> ARRAY_graph;
    ARRAY_graph = graph.toCSR();

    CD_ITERATION = graph.cdlp_max_its;
    CD_GRAPHSIZE = ARRAY_graph.OUTs_Neighbor_start_pointers.size() - 1;
    // CD_SET_THREAD = 100;

    outs_ptr.resize(CD_GRAPHSIZE + 1);
    outs_ptr = ARRAY_graph.OUTs_Neighbor_start_pointers;
    ins_ptr.resize(CD_GRAPHSIZE + 1);
    ins_ptr = ARRAY_graph.INs_Neighbor_start_pointers;
    in_out_ptr.resize(CD_GRAPHSIZE + 1);
    in_out_ptr[0] = 0;
    // int cnt=0;
    for (int i = 1; i <= CD_GRAPHSIZE; ++i)
    { 
        in_out_ptr[i] = in_out_ptr[i - 1] + (ins_ptr[i] - ins_ptr[i - 1]) + (outs_ptr[i] - outs_ptr[i - 1]);
                // if(in_out_ptr[i]==in_out_ptr[i-1]) cnt++;
    }
            // cout<<"zero num : "<<cnt<<endl;
    // for(int i=0;i<10000;++i){
    //     cout<<outs_ptr[i]<<"  "<<ins_ptr[i]<<"  "<<in_out_ptr[i]<<endl;
    // }
    outs_neighbor = ARRAY_graph.OUTs_Edges;
    ins_neighbor = ARRAY_graph.INs_Edges;
}

__global__ void init_label(int *labels_gpu, int *new_labels_gpu, int CD_GRAPHSIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < CD_GRAPHSIZE)
    {
        labels_gpu[tid] = tid;
        new_labels_gpu[tid] = tid;
    }
}

__global__ void extract_labels(int *in_out_ptr_gpu, int *ins_ptr_gpu, int *outs_ptr_gpu, int *ins_neighbor_gpu, int *outs_neighbor_gpu, int *labels, int *labels_out, int CD_GRAPHSIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= CD_GRAPHSIZE)
        return;

    int start = in_out_ptr_gpu[tid];
    int len_out = outs_ptr_gpu[tid + 1] - outs_ptr_gpu[tid];
    int len_in = ins_ptr_gpu[tid + 1] - ins_ptr_gpu[tid];
    // int end = in_out_ptr_gpu[tid + 1];
    for (int i = 0; i < len_in; ++i)
    {
        labels_out[start + i] = labels[ins_neighbor_gpu[ins_ptr_gpu[tid] + i]];
    }

    for (int i = 0; i < len_out; ++i)
    {
        labels_out[start + len_in + i] = labels[outs_neighbor_gpu[outs_ptr_gpu[tid] + i]];
    }
    // labels_out[start + len_in + len_out] = labels[tid];

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
    __syncthreads();
}

inline void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

// QuickSort function for raw pointers
void quickSort(int *arr, int low, int high)
{
    if (low < high)
    {
        int pivot = arr[high]; // choosing the last element as pivot
        int i = low - 1;

        for (int j = low; j < high; j++)
        {
            if (arr[j] < pivot)
            {
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


void bubbleSort(int *arr, int low, int high)
{
    for (int i = low; i < high; ++i)
    {
        for (int j = low; j < high - (i - low); ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

__global__ void LPA(int *global_space_for_label, int *in_out_ptr_gpu, int *labels_gpu, int *new_labels_gpu, int CD_GRAPHSIZE)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= CD_GRAPHSIZE)
        return;
    int start = in_out_ptr_gpu[tid], end = in_out_ptr_gpu[tid + 1];
    int current_label = global_space_for_label[start];
    int current_count = 1;
    int max_label = current_label;
    int max_count = current_count;
    for (int i = start+1; i < end; ++i)
    {
        if (global_space_for_label[i] == current_label)
        {
            current_count++;
        }
        else
        {
            if (current_count > max_count)
            {
                max_count = current_count;
                max_label = current_label;
            }
            else if (current_count == max_count && current_label < max_label)
            {
                max_label = current_label;
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
        max_label = current_label;
    }
    // if(max_label!=-1)
    new_labels_gpu[tid] = max_label;
    // else
    //     new_labels_gpu[tid] = labels_gpu[tid];
}

// void thrust_segmented_sort(int *array, vector<int> in_out_ptr, int CD_GRAPHSIZE)
// {
//     // 将所有数据一次性复制到设备向量
//     thrust::device_vector<int> d_array(array, array + in_out_ptr[CD_GRAPHSIZE]);

//     // 对每个段进行排序
//     for (int i = 0; i < CD_GRAPHSIZE; ++i)
//     {
//         int start = in_out_ptr[i];
//         int end = in_out_ptr[i + 1];
//         thrust::sort(thrust::device, d_array.begin() + start, d_array.begin() + end);
//     }

//     // 将排序后的数据一次性复制回主机数组
//     thrust::copy(d_array.begin(), d_array.end(), array);
// }


void thrust_segmented_sort(int *array, int *in_out_ptr, int CD_GRAPHSIZE)
{
    // Initialize CUB parameters
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;

    // 计算临时存储空间的大小
    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, array, array,
                                            in_out_ptr[CD_GRAPHSIZE], CD_GRAPHSIZE,
                                            in_out_ptr, in_out_ptr + 1);

    // 分配临时存储空间
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 执行分段排序
    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, array, array,
                                            in_out_ptr[CD_GRAPHSIZE], CD_GRAPHSIZE,
                                            in_out_ptr, in_out_ptr + 1);

    // 释放临时存储空间
    cudaFree(d_temp_storage);
}

int Community_Detection(graph_structure<double> &graph, float *elapsedTime, vector<int> &ans)
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
    cudaMallocManaged(&global_space_for_label, (outs_neighbor.size() + ins_neighbor.size()) * sizeof(int));
    cudaMallocManaged(&in_out_ptr_gpu, (CD_GRAPHSIZE + 1) * sizeof(int));

    cudaMemcpy(outs_ptr_gpu, outs_ptr.data(), outs_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ins_ptr_gpu, ins_ptr.data(), ins_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(outs_neighbor_gpu, outs_neighbor.data(), outs_neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ins_neighbor_gpu, ins_neighbor.data(), ins_neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_out_ptr_gpu, in_out_ptr.data(), in_out_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    // cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, (int*)nullptr, (int*)nullptr,
    //                                         in_out_ptr[CD_GRAPHSIZE], CD_GRAPHSIZE,
    //                                         in_out_ptr_gpu, in_out_ptr_gpu + 1);
    // cout<<"temp_storage_bytes : "<<temp_storage_bytes<<endl;
    // cudaMallocManaged(&d_temp_storage, temp_storage_bytes);
    // checkDeviceProperties();
    // get_size();
    // cout << 2 << endl;
    cudaError_t err;
    init_label<<<init_label_block, init_label_thread>>>(labels_gpu, new_labels_gpu, CD_GRAPHSIZE);
    err = cudaDeviceSynchronize();
    checkCudaError(err, "cudaDeviceSynchronize after init_label");
    // cout << 3 << endl;
    for(int i=0;i<CD_GRAPHSIZE;i++){
        if(labels_gpu[i]!=i) cout<<"error : "<<i<<endl;
    }
    // cudaEvent_t GPUstart, GPUstop;
    // cudaEventCreate(&GPUstart);
    // cudaEventCreate(&GPUstop);
    // cudaEventRecord(GPUstart, 0);
    
    double extract_labels_time = 0, LPA_time = 0, quickSort_time = 0;
    double start, mid, end;
    int it = 0;
    // cout << 4 << endl;

    // CD_ITERATION=3;
    // vector<int> threads(1000);
    omp_set_num_threads(100);
    // getchar();
    start = omp_get_wtime();
    while (it < CD_ITERATION)
    {
        if (it % 2 == 0)
        {
            // start = clock();
            extract_labels<<<init_label_block, init_label_thread>>>(in_out_ptr_gpu, ins_ptr_gpu, outs_ptr_gpu, ins_neighbor_gpu, outs_neighbor_gpu, labels_gpu, global_space_for_label, CD_GRAPHSIZE);
            err = cudaDeviceSynchronize();
            checkCudaError(err, "cudaDeviceSynchronize after extract_labels");
            
            // // mid = clock();
            // extract_labels_time += (mid - start);
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
            // start = clock();

#pragma omp parallel
            {
                // int num_threads = omp_get_num_threads();
                // int thread_id = omp_get_thread_num();
#pragma omp for
                for (int i = 0; i < CD_GRAPHSIZE; ++i)
                {
                    // std::cout << "Thread " << thread_id << " is processing index " << i << std::endl;
                    // threads[thread_id]++;

                    quickSort(global_space_for_label, in_out_ptr[i], in_out_ptr[i + 1] - 1);
                }
            }
            // thrust_segmented_sort(global_space_for_label,in_out_ptr_gpu,CD_GRAPHSIZE);
            // cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, global_space_for_label, global_space_for_label,
            //                                 in_out_ptr[CD_GRAPHSIZE], CD_GRAPHSIZE,
            //                                 in_out_ptr_gpu, in_out_ptr_gpu + 1);
            // mid = clock();
            // quickSort_time += mid - start;
            // int cnt=0;
            // for(int i=0;i<CD_GRAPHSIZE;++i){
            //     //cout<<ins_ptr[i+1]<<"  "<<ins_ptr[i]<<"  "<<outs_ptr[i+1]<<"  "<<outs_ptr[i]<<"  "<<ins_ptr[i+1]-ins_ptr[i]<<"  "<<outs_ptr[i+1]-outs_ptr[i]<<"  "<<in_out_ptr[i]<<"  "<<in_out_ptr[i+1]<<endl;
            //     bool f=false;
            //     for(int j=in_out_ptr[i]+1;j<in_out_ptr[i+1];j++){
            //         // cout<<global_space_for_label[j]<<"   ";
            //         if(global_space_for_label[j]<global_space_for_label[j-1]){
            //             cnt++;
            //             f=true;
            //             cout<<"error at ver : "<<i<<"  ";
            //         }
            //     }
            //     if(f) cout<<endl<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++   error num : "<<cnt<<endl;
            // }
            // start = clock();
            LPA<<<init_label_block, init_label_thread>>>(global_space_for_label, in_out_ptr_gpu, labels_gpu, new_labels_gpu, CD_GRAPHSIZE);
            err = cudaDeviceSynchronize();
            checkCudaError(err, "cudaDeviceSynchronize after LPA");
            // mid = clock();
            // LPA_time += mid - start;
        }
        else
        {
            // start = clock();
            extract_labels<<<init_label_block, init_label_thread>>>(in_out_ptr_gpu, ins_ptr_gpu, outs_ptr_gpu, ins_neighbor_gpu, outs_neighbor_gpu, new_labels_gpu, global_space_for_label, CD_GRAPHSIZE);
            err = cudaDeviceSynchronize();
            checkCudaError(err, "cudaDeviceSynchronize after extract_labels");
            // mid = clock();
            // extract_labels_time += (mid - start);
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
            // start = clock();
#pragma omp parallel
            {
                // int thread_id = omp_get_thread_num();
#pragma omp for
                for (int i = 0; i < CD_GRAPHSIZE; ++i)
                {
                    // std::cout << "Thread " << thread_id << " is processing index " << i << std::endl;
                    // threads[thread_id]++;
                    quickSort(global_space_for_label, in_out_ptr[i], in_out_ptr[i + 1] - 1);
                }
            }
            // thrust_segmented_sort(global_space_for_label,in_out_ptr_gpu,CD_GRAPHSIZE);
            // cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, global_space_for_label, global_space_for_label,
            //                                 in_out_ptr[CD_GRAPHSIZE], CD_GRAPHSIZE,
            //                                 in_out_ptr_gpu, in_out_ptr_gpu + 1);
            // mid = clock();
            // quickSort_time += mid - start;
            // int cnt=0;
            // for(int i=0;i<CD_GRAPHSIZE;++i){
            //     bool f=false;
            //     // cout<<ins_ptr[i+1]<<"  "<<ins_ptr[i]<<"  "<<outs_ptr[i+1]<<"  "<<outs_ptr[i]<<"  "<<ins_ptr[i+1]-ins_ptr[i]<<"  "<<outs_ptr[i+1]-outs_ptr[i]<<"  "<<in_out_ptr[i]<<"  "<<in_out_ptr[i+1]<<endl;
            //     for(int j=in_out_ptr[i]+1;j<in_out_ptr[i+1];j++){
            //         // cout<<global_space_for_label[j]<<"   ";
            //         if(global_space_for_label[j]<global_space_for_label[j-1]){
            //             f=true;
            //             cnt++;
            //             cout<<"error at ver : "<<i<<endl;
            //         }

            //     }
                
            //     if(f) cout<<endl<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++  error num : "<<cnt<<endl;
            // }
            // start = clock();
            LPA<<<init_label_block, init_label_thread>>>(global_space_for_label, in_out_ptr_gpu, new_labels_gpu, labels_gpu, CD_GRAPHSIZE);
            err = cudaDeviceSynchronize();
            checkCudaError(err, "cudaDeviceSynchronize after LPA");
            // mid = clock();
            // LPA_time += mid - start;
        }

        it++;
        cout<<"iteration : "<<it<<endl;
    }
    end = omp_get_wtime();
    // cout<<"threads : ";
    // for(int i=0;i<1000;i++){
    //     if(threads[i]!=0) cout<<i<<"  ";
    // }
    // cout<<endl;
    // cout << "extract_label_time : " << extract_labels_time / CLOCKS_PER_SEC * 1000 << " ms" << endl;
    // cout << "quickSort_time :  : " << quickSort_time / CLOCKS_PER_SEC * 1000 << " ms" << endl;
    // cout << "LPA_time : " << LPA_time / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    // cudaEventRecord(GPUstop, 0);
    // cudaEventSynchronize(GPUstop);

    // cudaEventElapsedTime(elapsedTime, GPUstart, GPUstop);
    // cudaEventDestroy(GPUstart);
    // cudaEventDestroy(GPUstop);
    *elapsedTime=end-start;
    ans.resize(CD_GRAPHSIZE);
    if (CD_ITERATION % 2 == 0)
    {
        cudaMemcpy(ans.data(), labels_gpu, CD_GRAPHSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    }
    else
    {
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

void cdlp_ldbc_check(graph_structure<double>& graph, std::vector<string>& cpu_res, int & is_pass,vector<int> & wrong_ver,set<int>& wrong_label) {
    
    int size = cpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of CDLP results is not equal to the number of vertices!" << std::endl;
        return;
    }


    std::string base_line_file = "../results/cit-Patents-CDLP";
    // remove the last two char

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0,wrong=0;
    std::string line;

    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];
        
        if (cpu_res[v_id] != tokens[1]) {
            // std::cout << "Baseline file and GPU CDLP results are not the same!" << std::endl;
            // std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            // std::cout << "CPU CDLP result: " << cpu_res[v_id] << std::endl;
            wrong++;
            wrong_ver.push_back(v_id);
            wrong_label.insert(graph.vertex_str_to_id[cpu_res[v_id]]);
            // base_line.close();
            // return;
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    cout<<"wrong "<<wrong<<"/"<<id<<"      "<<(double)wrong/id*100<<"%"<<endl;
    // std::cout << "CDLP results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}

int main()
{
    std::string config_file;
    std::cout << "Enter the name of the configuration file:" << std::endl;
    // std::cin >> config_file;
    config_file="cit-Patents.properties";
    config_file = "../data/" + config_file;

    graph_structure<double> graph;
    graph.read_config(config_file);

    graph.load_LDBC();
    CSR_graph<double> csr_graph = graph.toCSR();
    std::cout << "Number of vertices: " << csr_graph.OUTs_Neighbor_start_pointers.size() << std::endl;
    std::cout << "Number of edges: " << csr_graph.OUTs_Edges.size() << std::endl;

    float elapsedTime = 0;
    float cpu_time=0;
    float gpu_time=0;
    // clock_t start = clock(), end = clock();

    vector<int> ans_gpu;
    elapsedTime = 0;
    int flag;
    vector<int> wrong_ver;
    set<int> wrong_label;
    Community_Detection(graph, &elapsedTime,ans_gpu);
    vector<string> ans_string(CD_GRAPHSIZE);
    for(int i=0;i<CD_GRAPHSIZE;i++){
        ans_string[i]=graph.vertex_id_to_str[ans_gpu[i]];
    }
    cout<<"checking..."<<endl;
    cdlp_ldbc_check(graph,ans_string,flag,wrong_ver,wrong_label);
    int cnt_in=0,cnt_out=0,cnt_all=0;
    
   
    for(auto it:wrong_label)
    {   
        cout<<"wrong label : "<<it <<"   it's neighbor : ";
        // cout<<it<<" ";
        for(int i=ins_ptr[it];i<ins_ptr[it+1];++i){
            cout<<ins_neighbor[i]<<"  ";
        }
        if(ins_ptr[it]==ins_ptr[it+1]) cnt_in++;
        if(outs_ptr[it]==outs_ptr[it+1]) cnt_out++;
        
        cout<<"|  ";
        for(int i=outs_ptr[it];i<outs_ptr[it+1];++i){
            cout<<outs_neighbor[i]<<"  ";
        }
        cout<<endl;
    }
    cnt_in=0;cnt_out=0;cnt_all=0;
    for(int i=0;i<CD_GRAPHSIZE;++i){
        // if(ins_ptr[i]==ins_ptr[i+1]) cnt_in++;
        // if(outs_ptr[i]==outs_ptr[i+1]) cnt_out++;
        if(in_out_ptr[i]+1==in_out_ptr[i+1]) cnt_all++;
    }
    // cout<<cnt_in<<" "<<cnt_out<<endl;
    // // cout<<endl;
    // cout<<"cnt_in : "<<cnt_in<<"  cnt_out : "<<cnt_out<<endl;
    cout<<"cnt_all : "<<cnt_all<<endl;
    cout<<"wrong_label size : "<<wrong_label.size()<<endl;
    printf("GPU Community Detection cost time: %f s\n", elapsedTime);
}
