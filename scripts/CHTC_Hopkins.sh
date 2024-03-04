#!/bin/sh

cd ../src/

# Calculate the expression
# mr=$(expr $1 + 1)
mr=$(expr \( $1 % 9 \) + 1 )
idx=$(expr $1 / 9)

echo "MR:0.$mr"
echo "Hopkins_index:$idx"

python main.py --Hopkins_index $idx --missing_rate 0.$mr --step_method Regular --max_iter 1 --step_size 1 --dataset Hopkins155 --experiment_name CHTC_Hopkins155_ITER1k_MR0.$mr

cd ../scripts
