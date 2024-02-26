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



def convert_distance_to_similarity(d_matrix):
    # Convert distance matrix to similarity matrix using Gaussian kernel
    # The choice of gamma here is arbitrary; you might need to tune it for your specific case
    gamma = 1.0
    similarity_matrix = np.exp(-gamma * d_matrix ** 2)
    return similarity_matrix


def main(args, run_idx = 0):
    # Logging the hyperparams
    mlf_logger = MLFlowLogger(experiment_name=args.experiment_name, run_name = f"run_{run_idx}", save_dir = '../logs')
    mlf_logger.log_hyperparams(args)

    if args.dataset ==  'Synthetic':
        m = 100 #100-300
        n = 100 #100-300
        r = 3 #3-5
        K = 2
        if args.single_cluster:
            K = 1
            print('K = ', K)
        #all-in-one init function
        X_omega, labels, Omega, info = initialize_X_with_missing(m,n,r,K,args.missing_rate)
    else:
        raise ValueError(f"dataset {dataset} is not implemented!")

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
                            U_manifold_bound = 1e-5)


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
    parser.add_argument('--num_rums', type=int, default=1, help='number of runs')
    parser.add_argument('--step_method', type=str, default='Armijo', help='step_method')
    parser.add_argument('--lambda_in', type=float, default=1e-5, help='Lambda value (default: 1e-5)')
    parser.add_argument('--missing_rate', type=float, default=0, help='missing_rate (default: 1)')
    parser.add_argument('--max_iter', type=int, default=50, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--dataset', type=str, default='Synthetic', help='Dataset name (default: Synthetic)')
    parser.add_argument('--step_size', type=float, default=1, help='Step size (default: 1)')

    parser.add_argument('--single_cluster', action='store_true', default=False)



    # Parse the arguments
    args = parser.parse_args()

    for run_idx in range(args.num_rums):
        main(args, run_idx)
