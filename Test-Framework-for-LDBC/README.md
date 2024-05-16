# Test-Framework-for-LDBC
Based on graph algorithm library of RUC

## Environment

- Linux (Ubuntu 22.04+)
- CUDA 11.8+
- CMake 3.9+

## File Structure

- `src/`: source code
- `include/`: header files
- `data/`: put the LDBC dataset here (.properties, .v, .e)

## Build & Run

Fistly, you need to put the LDBC dataset in the `data/` directory.

For example, if you want to load the `cit-Patents` dataset, you should put the `cit-Patents.properties`, `cit-Patents.v`, `cit-Patents.e` in the `data/` directory.

Then, you can build and run the test program.

```shell
mkdir build
cd build
cmake ..
make
./bin/Test
```

Then the program will ask you to input the dataset configuration file name, you can input `cit-Patents.properties` to test the program.

```shell
Enter the name of the configuration file: # The program asking
cit-Patents.properties # Input the configuration file name
```

## Reminder

Please remove the `build/` directory and your dataset before you push your commit.