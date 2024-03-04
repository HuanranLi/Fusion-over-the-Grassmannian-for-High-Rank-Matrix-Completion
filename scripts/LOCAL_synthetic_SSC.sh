#!/bin/sh

cd ../src/

# Calculate the expression
mr=$(expr $1 + 1)

# Display the result
# echo "0.$mr"

python main.py --missing_rate 0.$mr --num_runs 1 --samples_per_class 50 --method ZF_SSC

cd ../scripts
