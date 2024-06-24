cd Test-Framework-for-LDBC
cp -r include CPU
cd CPU
rm -rf build
mkdir build
cd build
cmake ..
make
./bin/test-cpu