#!/bin/bash

# Define the number of times you want to run the script
number_of_runs=9
script="LOCAL_synthetic_K2.sh"

# Loop and call other_script.sh with index
for (( i=0; i<number_of_runs; i++ ))
do
   echo "Running iteration $i"
   ./$script $i
done

echo "All iterations completed."
