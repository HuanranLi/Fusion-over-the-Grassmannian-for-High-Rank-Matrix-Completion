#!/bin/sh

cd ../src/

# Calculate the expression
mr=$(expr \( $1 % 9 \) + 1 )
lambda=1e-3

# Display the result
echo "MR:0.$mr"
exp_name=L{$lambda}_CHTC_MNIST_EWZFSSC_0.$mr
echo "Experiment Name:"
echo $exp_name

python main.py --missing_rate 0.$mr --lambda_in $lambda --num_runs 1 --num_cluster 2 --samples_per_class 50 --method EWZF_SSC --dataset MNIST --experiment_name $exp_name

cd ../scripts
