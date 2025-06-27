# run chmod +x ./LDBC-GPU.sh to make this file executable

data_dir="/home/mabojing/data"

executable="/home/mdnd/CPU_GPU_project-main/build/bin_gpu/Test_GPU_CSR"

filenames=(
    "wiki-Talk"
    "cit-Patents"
    "kgs"
    "datagen-7_7-zf"
    "datagen-7_5-fb"
    "datagen-7_8-zf"
    "datagen-7_6-fb"
    "dota-league"
    "graph500-22"
    "datagen-7_9-fb"
    "datagen-8_2-zf"
    "datagen-8_0-fb"
    "graph500-23"
    "datagen-8_3-zf"
    "datagen-8_1-fb"
    "graph500-24"
    "datagen-8_4-fb"
    "datagen-8_5-fb"
    "datagen-8_7-zf"
    "datagen-8_6-fb"
    "datagen-8_8-zf"
    "graph500-25"
    "datagen-8_9-fb"
)

# 指定使用的GPU ID（例如：0, 1, 2）
gpu_id=1

for filename in "${filenames[@]}"
do
    for i in {1..1}
    do
        echo "testing file $filename ..."
        echo "$data_dir/" > input.txt
        echo "$filename" >> input.txt

        echo "$filename" >> result-gpu-csr.csv

        CUDA_VISIBLE_DEVICES=$gpu_id $executable < input.txt > "/home/mdnd/CPU_GPU_project-main/logs/$filename-GPU-csr.output"
        echo "test file $filename done."
    done
done