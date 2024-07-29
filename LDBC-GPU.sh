data_dir="/home/mabojing/data"

executable="/home/zhukai/CPU_GPU_project/build/bin_gpu/Test_GPU"

#filenames=("datagen-7_5-fb" "datagen-7_8-zf" "datagen-7_6-fb" "dota-league" "datagen-7_9-fb")
filenames=("kgs")

for filename in "${filenames[@]}"
do
    echo "testing file $filename ..."
    echo "$data_dir/$filename.properties" > input.txt
    echo "$data_dir/$filename.v" >> input.txt
    echo "$data_dir/$filename.e" >> input.txt
    echo "$data_dir/$filename-BFS" >> input.txt
    echo "$data_dir/$filename-SSSP" >> input.txt
    echo "$data_dir/$filename-WCC" >> input.txt
    echo "$data_dir/$filename-PR" >> input.txt
    echo "$data_dir/$filename-CDLP" >> input.txt

    $executable < input.txt > "/home/zhukai/CPU_GPU_project/results/$filename-GPU.output"
    echo "test file $filename done."
done