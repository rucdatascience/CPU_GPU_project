cd Test-Framework-for-LDBC
cd CPU
rm -rf include
cd ..
rm -rf build
mkdir build
cd build
cmake ..
make
./bin/Test
