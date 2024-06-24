# Test-Framework-for-LDBC
Based on graph algorithm library of RUC

## Environment

- Linux (Ubuntu 22.04+)
- CUDA 11.8+
- CMake 3.9+

## File Structure

- `src/`: source code
- `include/`: header files.**Please note that there is no include folder in the current folder, you need to copy the include from Test-Framework-for-LDBC to the CPU directory. If you want to run the entire project, execute the test.sh file in the Test-Framework-for-LDBC directory and delete the include folder in the CPU directory** 
- `data/`: put the LDBC dataset here (.properties, .v, .e)
- `results/`: put the baseline results here, the program will check the algorithm results automatically

## Build & Run

Fistly, you need to put the LDBC dataset in the `data/` directory.

For example, if you want to load the `cit-Patents` dataset, you should put the `cit-Patents.properties`, `cit-Patents.v`, `cit-Patents.e` in the `data/` directory.

Then, you can build and run the test program.

```shell
mkdir build
cd build
cmake ..
make
./bin/test-cpu
```
You can also use the simpler quick test below.

```shell
cd CPU_GPU_project/Test-Framework-for-LDBC/CPU
chmod +x test-cpu.sh
./test-cpu.sh
```

or
```shell
cd CPU_GPU_project
./cpu-test.sh
```

Then the program will ask you to input the dataset configuration file name, you can input `cit-Patents.properties` to test the program.

```shell
Enter the name of the configuration file: # The program asking
cit-Patents.properties # Input the configuration file name
```

## Reminder

Please remove the `build/` directory and your dataset before you push your commit.