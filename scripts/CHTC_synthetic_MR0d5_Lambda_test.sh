#!/bin/sh

cd ../src/

# Calculate the expression
# mr=$(expr $1 + 1)

# Display the result
# echo "0.$mr"

python main.py --missing_rate 0.5 --step_method Regular --max_iter 5000 --step_size 1 --num_rums 3 --lambda_in 1e-$1 --experiment_name CHTC_ITER10K_Synthetic_LambdaTest_1e-$1

cd ../scripts
