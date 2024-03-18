#!/bin/sh

cd ../src/

mr=$(expr \( $1 % 9 \) + 1 )
# Display the result
echo "0.$mr"

python main.py --check_acc_per_iter 300 --experiment_name CHTC_VIS_ITER30K_MR0.$mr --missing_rate 0.$mr --step_method Regular --max_iter 30000 --step_size 1 --num_runs 1  --samples_per_class 50 --num_cluster 2

cd ../scripts
