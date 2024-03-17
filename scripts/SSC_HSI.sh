#!/bin/sh

cd ../src/

# Calculate the expression
# mr=$(expr $1 + 1)
mr=$(expr \( $1 % 9 \) + 1 )
idx=$(expr \( $1 / 9 \) % 5 )
lambda=1e-3

echo "MR:0.$mr"
echo "HSI_idx:$idx"
echo "lambda:$lambda"
echo

python main.py --missing_rate 0.$mr --lambda_in $lambda --num_runs 1 --num_cluster 2 --samples_per_class 50 --method EWZF_SSC --dataset HSI_$idx --experiment_name Hopkins155_EWZFSSC_0.$mr

cd ../scripts
