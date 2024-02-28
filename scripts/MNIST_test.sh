cd ../src/

python main.py --check_acc_per_iter 5 --missing_rate 0.3 --step_method Regular --max_iter 20 --step_size 1 --num_cluster 2 --samples_per_class 50 --dataset MNIST --experiment_name MNIST_LOCAL

cd ../scripts
