# Test-Framework-for-LDBC
Based on graph algorithm library of RUC

## Environment

- Linux (Ubuntu 22.04+)
- CUDA 11.8
- CMake 3.9+

## File Structure

- `src_cpu/`: CPU source files
- `src_gpu/`: GPU source files
- `include/`: global header files
- `include_cpu/`: CPU header files
- `include_gpu/`: GPU header files
- `data/`: put the LDBC dataset here (.properties, .v, .e)
- `results/`: put the baseline results here, the program will check the algorithm results automatically

## Build & Run

Fistly, you need to put the LDBC dataset in the `data/` directory.

For example, if you want to load the `cit-Patents` dataset, you should put the `cit-Patents.properties`, `cit-Patents.v`, `cit-Patents.e` in the `data/` directory.

Then, you can build and run the test program.

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

Then the program will ask you to input the dataset configuration file name, you can input `cit-Patents.properties` to test the program.

```shell
Enter the name of the configuration file: # The program asking
cit-Patents.properties # Input the configuration file name
```
