cd Test-Framework-for-LDBC/GPU_CSR
rm -rf build
mkdir build
cd build
cmake ..
make
./bin/Test
