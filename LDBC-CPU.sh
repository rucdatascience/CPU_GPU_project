# run chmod +x ./LDBC-CPU.sh to make this file executable

data_dir="/home/jongmelon/LDBC-data"

executable="/home/jongmelon/CPU_GPU_project/build/bin_cpu/Test_CPU"

filenames=("datagen-7_5-fb")

for filename in "${filenames[@]}"
do
    echo "testing file $filename ..."
    echo "$data_dir/" > input.txt
    echo "$filename" >> input.txt

    $executable < input.txt > "/home/jongmelon/CPU_GPU_project/logs/$filename-CPU.output"
    echo "test file $filename done."
done