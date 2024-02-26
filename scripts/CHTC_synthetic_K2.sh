#!/bin/sh

cd ../src/

# Calculate the expression
mr=$(expr $1 + 1)

# Display the result
# echo "0.$mr"

python main.py --distance_to_truth --missing_rate 0.$mr --step_method Regular --max_iter 5000 --step_size 1 --num_rums 1 --lambda_in 0.001 --experiment_name CHTC_ITER5k_Synthetic_K2L3_MR0.$mr --run_name MR0.$mr

cd ../scripts
