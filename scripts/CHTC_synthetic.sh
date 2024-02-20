#!/bin/sh

cd ../src/

# Calculate the expression
mr=$(expr $1 + 0)

# Display the result
echo "0.$mr"

python main.py --missing_rate 0.$mr --step_method Regular --max_iter 2000 --step_size 1 --num_rums 3 --experiment_name Synthetic_MR0.$mr

cd ../scripts
