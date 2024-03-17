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

python main.py --lambda_in $lambda --missing_rate 0.$mr --step_method Regular --max_iter 12001 --step_size 1 --dataset HSI_$idx --experiment_name CHTC_HSI_12K_L{$lambda}_MR0.$mr --check_acc_per_iter 500 --num_cluster 2 --samples_per_class 50

cd ../scripts
