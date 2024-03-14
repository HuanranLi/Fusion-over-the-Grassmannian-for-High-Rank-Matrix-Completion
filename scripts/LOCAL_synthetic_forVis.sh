#!/bin/sh

cd ../src/


# Display the result
# echo "0.$mr"

python main.py --check_acc_per_iter 20 --experiment_name visualization --missing_rate 0.7 --step_method Regular --max_iter 1000 --step_size 1 --num_runs 1  --samples_per_class 25 --num_cluster 2

cd ../scripts
