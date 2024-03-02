#!/bin/sh

cd ../src/

# Calculate the expression
# mr=$(expr $1 + 1)
mr=$(expr \( $1 % 9 \) + 1 )

echo "MR:0.$mr"

python main.py --missing_rate 0.$mr --step_method Regular --max_iter 1000 --num_runs 1 --step_size 1 --num_cluster 2 --samples_per_class 50 --dataset MNIST --experiment_name CHTC_MNIST_0.$mr

cd ../scripts
