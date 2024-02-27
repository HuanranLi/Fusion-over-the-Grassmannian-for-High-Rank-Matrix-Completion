#!/bin/sh

cd ../src/

# Calculate the expression
mr=$(expr $1 + 1)

python main.py --missing_rate 0.$mr --step_method Regular --max_iter 100 --num_runs 3 --step_size 1 --num_cluster 2 --samples_per_class 50 --dataset MNIST --experiment_name CHTC_MNIST

cd ../scripts
