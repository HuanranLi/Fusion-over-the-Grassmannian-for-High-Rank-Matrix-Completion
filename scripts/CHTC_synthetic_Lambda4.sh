#!/bin/sh

cd ../src/

# Calculate the expression
mr=$(expr $1 + 1)

# Display the result
# echo "0.$mr"

python main.py --missing_rate 0.$mr --step_method Regular --max_iter 10000 --step_size 1 --num_rums 3 --lambda_in 0.0001 --experiment_name CHTC_ITER10K_Synthetic_Lambda1e-4_MR0.$mr

cd ../scripts
