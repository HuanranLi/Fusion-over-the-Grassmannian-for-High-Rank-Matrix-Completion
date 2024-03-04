#!/bin/sh

cd ../src/

# Calculate the expression
mr=$(expr \( $1 % 9 \) + 1 )

# Display the result
echo "MR:0.$mr"

python main.py --missing_rate 0.$mr --num_runs 1 --num_cluster 2 --samples_per_class 50 --method ZF_SSC --dataset MNIST --experiment_name CHTC_MNIST_ZFSSC_0.$mr

cd ../scripts
