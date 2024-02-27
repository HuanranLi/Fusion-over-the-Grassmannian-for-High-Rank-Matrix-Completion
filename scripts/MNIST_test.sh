cd ../src/

python main.py --missing_rate 0.2 --step_method Regular --max_iter 50 --step_size 1 --num_cluster 2 --samples_per_class 50 --dataset MNIST --experiment_name MNIST_LOCAL

cd ../scripts
