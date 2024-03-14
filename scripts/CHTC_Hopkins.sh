#!/bin/sh

cd ../src/

# Calculate the expression
# mr=$(expr $1 + 1)
mr=$(expr \( $1 % 6 \) )
idx=$(expr $1 / 6)

echo "MR:0.$mr"
echo "Hopkins_index:$idx"

python main.py --missing_rate 0.$mr --step_method Regular --max_iter 12001 --step_size 1 --dataset Hopkins155_$idx --experiment_name Hopkins155_ITER12k_MR0.$mr --check_acc_per_iter 500

cd ../scripts
