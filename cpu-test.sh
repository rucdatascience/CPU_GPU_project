cd Test-Framework-for-LDBC/CPU
rm -rf build
mkdir build
cd build
cmake ..
make
./bin/test-cpu