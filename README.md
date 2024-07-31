# Test-Framework-for-LDBC
Based on graph algorithm library of RUC

## Environment

- Linux (Ubuntu 22.04+)
- CUDA 11.8
- CMake 3.9+

## File Structure

- `include/`: header files
- `src/CPU_adj_list`: example of CPU version
- `src/GPU_csr`: example of GPU version
- `src/LDBC`: test framework for LDBC dataset

## Build & Run

By specifying compilation options, you can choose to compile the CPU version or the GPU version, or both.

```shell
mkdir build
cd build
cmake .. -DBUILD_CPU=OFF -DBUILD_GPU=ON # Compile the GPU version only
make
./bin_gpu/Test_GPU
```

By default, the program will compile the CPU version only.

```shell
mkdir build
cd build
cmake .. # Compile the CPU version only by default
make
./bin_cpu/Test_CPU
```

Of course, you can compile both the CPU version and the GPU version, if you have both the CPU and the GPU environment.

```shell
mkdir build
cd build
cmake .. -DBUILD_CPU=ON -DBUILD_GPU=ON # Compile both the CPU version and the GPU version
make
./bin_cpu/Test_CPU
./bin_gpu/Test_GPU
```

Then the program will ask you to input the directory and the graph name,

Note that the directory should contain the configuration file, the graph data, and the necessary baseline results.

```shell
Please input the data directory: # The program asking
/somewhere/ # Input the configuration file name
Please input the graph name: # The program asking
datagen-7_5-fb # Input the graph name which you want to test
```
