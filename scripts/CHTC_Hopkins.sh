#!/bin/sh

cd ../src/

# Calculate the expression
# mr=$(expr $1 + 1)
mr=$(expr \( $1 % 6 \) )
idx=$(expr $1 / 6)

echo "MR:0.$mr"
echo "Hopkins_index:$idx"
echo "python main.py --Hopkins_index $idx --missing_rate 0.$mr --step_method Regular --max_iter 3000 --step_size 1 --dataset Hopkins155 --experiment_name CHTC_Full_Hopkins155_ITER3k_V2_MR0.$mr --check_acc_per_iter 100
"

python main.py --Hopkins_index $idx --missing_rate 0.$mr --step_method Regular --max_iter 3000 --step_size 1 --dataset Hopkins155 --experiment_name CHTC_Full_Hopkins155_ITER3k_V2_MR0.$mr --check_acc_per_iter 100

cd ../scripts
