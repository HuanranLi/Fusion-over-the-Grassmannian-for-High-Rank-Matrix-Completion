from GrassmannianFusion import GrassmannianFusion
from Initialization import *
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
import argparse
# from PIL import Image
import io

from pytorch_lightning.loggers import MLFlowLogger

from sklearn import metrics
from helper_functions import *
from distances import *
from MNIST import *
from Hopkins155 import *

import os
# from multiprocessing import Pool
from SSC import *
from HSI import *
from callbacks import *
import re

def extract_index_from_prefix(input_string, prefix):
    match = re.search(rf'{prefix}(\d+)', input_string)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("No valid index found in the input string")



def main(args, run_idx = 0):
    # Logging the hyperparams
    log_dir = '../logs'
    run_name = args.run_name if args.run_name else f'run_{run_idx}'
    mlf_logger = MLFlowLogger(experiment_name=args.experiment_name, run_name = run_name, save_dir = log_dir)


    mlf_logger.log_hyperparams(args)
    callbacks = []
    K = args.num_cluster

    if args.dataset ==  'Synthetic':
        m = 100 #100-300
        n = args.samples_per_class * args.num_cluster #100-300
        r = 5 #3-5
        X_omega, labels, Omega, info = initialize_X_with_missing(m,n,r,K,args.missing_rate)
        true_subspaces = info['X_lowRank_array']
    elif args.dataset == 'MNIST':
        # Randomly select class indices
        class_indices = np.random.choice(10, args.num_cluster, replace=False)

        # Call the MNIST function
        X, labels = MNIST(class_indices, args.samples_per_class)
        X_omega, Omega = random_sampling(X, args.missing_rate)

        r = 3
        m, n = X.shape
    elif args.dataset.startswith('Hopkins155'):
        print('Warning: Hopkins155 is used! args.num_cluster, args.samples_per_class is not used in this case. Please check hyperparam m,n,K for the actual shape of data.')
        # Usage example:

        dataset_idx = extract_index_from_prefix(args.dataset, 'Hopkins155_')
        print('Current Dataset Index:', dataset_idx)

        hopkins_path = '../datasets/Hopkins155'
        # x_processed, s_array = process_mat_data(os.path.join(hopkins_path, trail_name, trail_name + "_truth.mat"))
        X, labels = process_hopkins_sequence(hopkins_path, index = dataset_idx)
        X_omega, Omega = random_sampling(X, args.missing_rate)

        r = 3
        m, n = X.shape
        K = len(set(labels))

    elif args.dataset.startswith('HSI'):
        dataset_idx = extract_index_from_prefix(args.dataset, 'HSI_')
        print('Current Dataset Index:', dataset_idx)

        # Call the MNIST function
        X, labels = load_sampled_hsi_data(dataset_idx, args.num_cluster, args.samples_per_class)
        X_omega, Omega = random_sampling(X, args.missing_rate)

        r = 3
        m, n = X.shape
    else:
        raise ValueError(f"dataset {dataset} is not implemented!")

    if args.distance_to_truth:
        callbacks.append(lambda instance: distance_to_truth_callback(instance, true_subspaces, labels) )
    if args.check_acc_per_iter:
        callbacks.append(lambda instance: check_clustering_acc(instance, labels,  args.check_acc_per_iter, K))



    mlf_logger.log_hyperparams({'m':m, 'n':n, 'r':r, 'K': K})
    print('Paramter: lambda = ',args.lambda_in,', K = ',K,', m = ', m, ', n = ',n,', r = ',r,', missing_rate =', args.missing_rate)

    if args.method == 'GF':
        #object init
        GF = GrassmannianFusion(X = X_omega,
                                Omega = Omega,
                                r = r,
                                lamb = args.lambda_in,
                                g_threshold= 1e-6,
                                bound_zero = 1e-10,
                                singular_value_bound = 1e-5,
                                g_column_norm_bound = 1e-5,
                                U_manifold_bound = 1e-5,
                                callbacks = callbacks)


        GF.train(max_iter = args.max_iter, step_size = args.step_size, logger = mlf_logger, step_method = args.step_method, multiprocessing = args.multiprocessing)
        d_matrix = GF.distance_matrix()
        similarity_matrix = convert_distance_to_similarity(d_matrix)
    elif args.method == 'ZF_SSC':
        similarity_matrix = zf_ssc(X_omega, Omega, lambda_val = args.lambda_in)
    elif args.method == "EWZF_SSC":
        similarity_matrix = ewzf_ssc(X_omega, Omega, lambda_val = args.lambda_in)


    pred_labels, metrics = spectral_clustering(similarity_matrix, K, labels)

    print(metrics)
    mlf_logger.log_metrics((metrics), step = args.max_iter)
    # plot_distance(d_matrix, labels, mlf_logger)



if __name__ == '__main__':
    # freeze_support()
    print('Starting Main')

    # Create the parser
    parser = argparse.ArgumentParser(description='Parameter settings for training')

    parser.add_argument('--method', type=str, default='GF', help='name of the method')

    # Add arguments
    parser.add_argument('--experiment_name', type=str, default='test', help='experiment_name')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--run_name', type=str, default=None, help='run_name')
    parser.add_argument('--step_method', type=str, default='Armijo', help='step_method')
    parser.add_argument('--lambda_in', type=float, default=1e-5, help='Lambda value (default: 1e-5)')
    parser.add_argument('--missing_rate', type=float, default=0, help='missing_rate (default: 1)')
    parser.add_argument('--max_iter', type=int, default=50, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--dataset', type=str, default='Synthetic', help='Dataset name (default: Synthetic)')
    parser.add_argument('--step_size', type=float, default=1, help='Step size (default: 1)')

    parser.add_argument('--num_cluster', type=int, default=2, help='number of clusters')
    parser.add_argument('--distance_to_truth', action='store_true', default=False)
    parser.add_argument("--samples_per_class", type=int, default=50, help="Number of images per class")
    parser.add_argument('--check_acc_per_iter', type=int, default=None, help='Check clustering accuracy per x iterations')

    parser.add_argument('--multiprocessing', action='store_true', default=False)




    # Parse the arguments
    args = parser.parse_args()

    print('Finish Parsing args.')
    for run_idx in range(args.num_runs):
        main(args, run_idx)
