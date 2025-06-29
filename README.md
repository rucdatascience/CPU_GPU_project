# RucGraph - a fast graph database system on CPU/GPU platforms

RucGraph is a lightweight graph database system that uses both CPUs and GPUs to efficiently perform graph analyses, such as Shortest Path, PageRank, Community Detection etc.


- "Ruc" is the abbreviation of "[Renmin University of China](https://www.ruc.edu.cn/)".


- RucGraph works efficiently on large graphs with billions of vertices and edges. In particular, on [LDBC Graphalytics Benchmarks](https://ldbcouncil.org/benchmarks/graphalytics/), RucGraph is <b>10 times faster than [neo4j](https://neo4j.com) on CPUs</b>, and <b>50 times faster than  [neo4j](https://neo4j.com) on GPUs</b>.




## Graph data structures & algorithms

RucGraph is now using [Adjacency Lists](https://www.geeksforgeeks.org/adjacency-list-meaning-definition-in-dsa/) to store graphs in CPU memory, and using [Sparse Matrix Representations](https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/) (CSRs), GPU Packed Memory Array (GPMA) as well as Adjacency Lists to store graphs in GPU memory. 


We have implemented 5 graph analysis algorithms on both CPUs and GPUs to date: Breadth-First Search (BFS), PageRank (PR), Weakly Connected Components (WCC), Community Detection using Label Propagation (CDLP), Single-Source Shortest Paths (SSSP). The pseudo codes of these algorithms can be found in the end of [the LDBC Graphalytics Benchmark handbook](https://arxiv.org/pdf/2011.15028). Nevertheless, our implementations are optimized for parallel computation, and may be considerably different from these pseudo codes.


## Code File structures

- `include/`: header files

- `include/CPU_adj_list/`: header files for operating **Adjacency Lists** on CPUs

- `include/CPU_adj_list/CPU_adj_list.hpp`: An Adjacency List on CPUs

- `include/CPU_adj_list/algorithm/`: header files for graph analysis operators on CPUs, such as Shortest Path, PageRank, Community Detection operators; these operators have passed the LDBC Graphalytics Benchmark test



- `include/GPU_adj_list/`: header files for operating **Adjacency Lists** on GPUs

- `include/GPU_adj_list/GPU_adj.hpp`: An Adjacency List on GPUs

- `include/GPU_adj_list/algorithm/`: header files for graph analysis operators on GPUs, such as Shortest Path, PageRank, Community Detection operators; these operators have also passed the LDBC Graphalytics Benchmark test

  

- `include/GPU_csr/`: header files for operating **CSRs** on GPUs

- `include/GPU_csr/GPU_csr.hpp`: A CSR on GPUs

- `include/GPU_csr/algorithm/`: header files for graph analysis operators on GPUs, such as Shortest Path, PageRank, Community Detection operators; these operators have also passed the LDBC Graphalytics Benchmark test

  

- `include/GPU_gpma/`: header files for operating **GPMA** on GPUs

- `include/GPU_gpma/GPU_gpma.hpp`: A GPU Packed Memory Array on GPUs

- `include/GPU_gpma/algorithm/`: header files for graph analysis operators on GPUs, such as Shortest Path, PageRank, Community Detection operators; these operators have also passed the LDBC Graphalytics Benchmark test


- `include/LDBC/`: header files for performing the LDBC Graphalytics Benchmark test

 <br />


- `src/`: source files
- `src/CPU_adj_list/CPU_example.cpp`: an example of performing graph analysis operators on CPUs
- `src/GPU_adj_list/GPU_adj_example.cu`: an example of performing graph analysis operators on GPUs (Adjacency List version)
- `src/GPU_csr/GPU_csr_example.cu`: an example of performing graph analysis operators on GPUs (CSR version)
- `src/GPU_gpma/GPU_gpma_example.cu`: an example of performing graph analysis operators on GPUs (GPMA version)
- `src/LDBC/LDBC_CPU_adj_list.cpp`: the source codes of performing the LDBC Graphalytics Benchmark test on CPUs
- `src/LDBC/LDBC_GPU_adj.cu`: the source codes of performing the LDBC Graphalytics Benchmark test on GPUs (Adjacency List version)
- `src/LDBC/LDBC_GPU_csr.cu`: the source codes of performing the LDBC Graphalytics Benchmark test on GPUs (CSR version)
- `src/LDBC/LDBC_GPU_gpma.cu`: the source codes of performing the LDBC Graphalytics Benchmark test on GPUs (GPMA version)



## Build & Run

Here, we show how to build & run RucGraph on a Linux server with the Ubuntu 20.04 system, 2 Intel(R) Xeon(R) Gold 5218 CPUs, and 4 NVIDIA GeForce RTX 3090 GPUs. The environment is as follows.

<b>On 170 server, before compiling, 1st, use the "source /opt/rh/devtoolset-11/enable" command to change g++; 2nd, change "cmake" in the following commands to "cmake3". </b>

- `cmake --version`: cmake version 3.27.9
- `g++ --version`: g++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
- `nvidia-smi`: NVIDIA-SMI 550.54.14         /      Driver Version: 550.54.14   /   CUDA Version: 12.4


First, download the files onto the server, e.g., onto the following path: `/home/username/RucGraph`. Second, enter the following commands on a terminal at this path:

```shell
username@server:~/RucGraph$ mkdir build
username@server:~/RucGraph$ cd build
username@server:~/RucGraph/build$ cmake .. -DBUILD_CPU=ON -DBUILD_GPU_ADJ=ON -DBUILD_GPU_CSR=ON -DBUILD_GPU_GPMA=ON
username@server:~/RucGraph/build$ make
username@server:~/RucGraph/build$ ./bin_cpu/CPU_example
username@server:~/RucGraph/build$ ./bin_gpu/GPU_example_adj
username@server:~/RucGraph/build$ ./bin_gpu/GPU_example_csr
username@server:~/RucGraph/build$ ./bin_gpu/GPU_example_gpma
username@server:~/RucGraph/build$ ./bin_cpu/Test_CPU
username@server:~/RucGraph/build$ ./bin_gpu/Test_GPU_ADJ
username@server:~/RucGraph/build$ ./bin_gpu/Test_GPU_CSR
username@server:~/RucGraph/build$ ./bin_gpu/Test_GPU_GPMA
```

There are some explanations for the above commands:

- `-DBUILD_CPU=ON -DBUILD_GPU_ADJ=ON -DBUILD_GPU_CSR=ON -DBUILD_GPU_GPMA=ON` is to compile both CPU, GPU (Adjacency List version), GPU (CSR version) and GPU (GPMA version) codes. If GPUs are not available, then we can change `-DBUILD_GPU_ADJ=ON -DBUILD_GPU_CSR=ON -DBUILD_GPU_GPMA=ON` to `-DBUILD_GPU_ADJ=OFF -DBUILD_GPU_CSR=OFF -DBUILD_GPU_GPMA=OFF`.


- `./bin_cpu/CPU_example` is to run the source codes at `src/CPU_adj_list/CPU_example.cpp`
- `./bin_gpu/GPU_example_adj` is to run the source codes at `src/GPU_adj_list/GPU_adj_example.cu`
- `./bin_gpu/GPU_example_csr` is to run the source codes at `src/GPU_csr/GPU_csr_example.cu`
- `./bin_gpu/GPU_example_gpma` is to run the source codes at `src/GPU_gpma/GPU_gpma_example.cu`
- `./bin_cpu/Test_CPU` is to run the source codes at `src/LDBC/LDBC_CPU_adj_list.cpp`
- `./bin_gpu/Test_GPU_ADJ` is to run the source codes at `src/LDBC/LDBC_GPU_adj.cu`
- `./bin_gpu/Test_GPU_CSR` is to run the source codes at `src/LDBC/LDBC_GPU_csr.cu`
- `./bin_gpu/Test_GPU_GPMA` is to run the source codes at `src/LDBC/LDBC_GPU_gpma.cu`

We can run "CPU_example", "GPU_example_adj", "GPU_example_csr" and "GPU_example_gpma" without any graph dataset. The outputs of graph analysis operators will be printed on the terminal. 

Nevertheless, before running "Test_CPU", "Test_GPU_ADJ", "Test_GPU_CSR" and "Test_GPU_GPMA", we need to download the [LDBC Graphalytics datasets](https://repository.surfsara.nl/datasets/cwi/graphalytics) at first. Then, when running "Test_CPU", "Test_GPU_ADJ", "Test_GPU_CSR" and "Test_GPU_GPMA", the program will ask us to input the data path and name sequentially. 

```shell
Please input the data directory: # The program asks
/home/username/data # Input the data path
Please input the graph name: # The program asks
datagen-7_5-fb # Input a data name
```

After inputting the data path and name, the program will perform the LDBC Graphalytics Benchmark test for this dataset. Specifically, the program will print some parameters of this test, as well as the consumed times of different graph analysis operators on this dataset.