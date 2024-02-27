from GrassmannianFusion import GrassmannianFusion
from Initialization import *
from helper_functions import evaluate

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import argparse
from PIL import Image
import io

from pytorch_lightning.loggers import MLFlowLogger

from sklearn import metrics
from helper_functions import *
from distances import *
from MNIST import *



def convert_distance_to_similarity(d_matrix):
    # Convert distance matrix to similarity matrix using Gaussian kernel
    # The choice of gamma here is arbitrary; you might need to tune it for your specific case
    gamma = 1.0
    similarity_matrix = np.exp(-gamma * d_matrix ** 2)
    return similarity_matrix


def distance_to_truth_callback(instance, truth_subspaces, labels):
    distances = np.mean([geodesic(U, truth_subspaces[labels[i]]) for i,U in enumerate(instance.U_array)])
    instance.logger.log_metrics(({'distance_to_truth_mean': distances}), step=instance.iter)



def main(args, run_idx = 0):
    # Logging the hyperparams
    run_name = args.run_name if args.run_name else f'run_{run_idx}'
    mlf_logger = MLFlowLogger(experiment_name=args.experiment_name, run_name = run_name, save_dir = '../logs')
    mlf_logger.log_hyperparams(args)
    callbacks = []

    K = args.num_cluster

    if args.dataset ==  'Synthetic':
        m = 100 #100-300
        n = args.samples_per_class * args.num_cluster #100-300
        r = 3 #3-5
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

    else:
        raise ValueError(f"dataset {dataset} is not implemented!")

    if args.distance_to_truth:
        callbacks.append(lambda instance: distance_to_truth_callback(instance, true_subspaces, labels) )



    mlf_logger.log_hyperparams({'m':m, 'n':n, 'r':r, 'K': K})
    print('Paramter: lambda = ',args.lambda_in,', K = ',K,', m = ', m, ', n = ',n,', r = ',r,', missing_rate =', args.missing_rate)

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


    GF.train(max_iter = args.max_iter, step_size = args.step_size, logger = mlf_logger, step_method = args.step_method)
    d_matrix = GF.distance_matrix()
    similarity_matrix = convert_distance_to_similarity(d_matrix)
    pred_labels, metrics = spectral_clustering(similarity_matrix, K, labels)

    print(metrics)
    mlf_logger.log_metrics((metrics))
    plot_distance(d_matrix, labels, mlf_logger)



if __name__ == '__main__':

        # Create the parser
    parser = argparse.ArgumentParser(description='Parameter settings for training')

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




    # Parse the arguments
    args = parser.parse_args()

    for run_idx in range(args.num_runs):
        main(args, run_idx)
