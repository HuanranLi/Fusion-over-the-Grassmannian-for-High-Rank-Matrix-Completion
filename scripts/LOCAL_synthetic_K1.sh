#!/bin/sh

cd ../src/

# Calculate the expression
mr=$(expr $1 + 1)

# Display the result
# echo "0.$mr"

python main.py --single_cluster --distance_to_truth --missing_rate 0.$mr --step_method Regular --max_iter 50 --step_size 1 --num_rums 1 --lambda_in 0.001 --experiment_name LOCAL_ITER50_Synthetic_K1L3_MR0.$mr

cd ../scripts
