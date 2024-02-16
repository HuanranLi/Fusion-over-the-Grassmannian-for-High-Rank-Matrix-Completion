from GrassmannianFusion import GrassmannianFusion
from Initialization import *
from helper_functions import evaluate

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import argparse
import seaborn as sns

def convert_distance_to_similarity(d_matrix):
    # Convert distance matrix to similarity matrix using Gaussian kernel
    # The choice of gamma here is arbitrary; you might need to tune it for your specific case
    gamma = 1.0
    similarity_matrix = np.exp(-gamma * d_matrix ** 2)
    return similarity_matrix



def main(args):

    if args.dataset ==  'Synthetic':
        m = 50 #100-300
        n = 50 #100-300
        r = 3 #3-5
        K = 3
        init_params = (m,n,r,K,args.missing_rate)
        #all-in-one init function
        X_omega, labels, Omega, info = initialize_X_with_missing(init_params)


        print('Paramter: lambda = ',args.lambda_in,', K = ',K,', m = ', m, ', n = ',n,', r = ',r,', missing_rate =', args.missing_rate)

    #object init
    GF = GrassmannianFusion(X = X_omega,
                            Omega = Omega,
                            r = r,
                            lamb = args.lambda_in,
                            weight_factor = args.weight_f_in,
                            g_threshold= 1e-6,
                            bound_zero = 1e-10,
                            singular_value_bound = 1e-5,
                            g_column_norm_bound = 1e-5,
                            U_manifold_bound = 1e-5)


    GF.train(max_iter = args.max_iter, step_size = args.step_size)
    d_matrix = GF.distance_matrix()
    similarity_matrix = convert_distance_to_similarity(d_matrix)


    # Perform Spectral Clustering
    sc = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=0)
    clusters = sc.fit_predict(similarity_matrix)

    print(clusters)
    print('SC Accuracy:' , 1 - evaluate(clusters, labels , K))



if __name__ == '__main__':

        # Create the parser
    parser = argparse.ArgumentParser(description='Parameter settings for training')

    # Add arguments
    parser.add_argument('--lambda_in', type=float, default=1, help='Lambda value (default: 1)')
    parser.add_argument('--weight_f_in', type=float, default=1, help='Weight factor (default: 1)')
    parser.add_argument('--missing_rate', type=float, default=0, help='missing_rate (default: 1)')
    parser.add_argument('--max_iter', type=int, default=50, help='Maximum number of iterations (default: 50)')
    parser.add_argument('--dataset', type=str, default='Synthetic', help='Dataset name (default: Synthetic)')
    parser.add_argument('--step_size', type=float, default=1, help='Step size (default: 1)')

    # Parse the arguments
    args = parser.parse_args()

    main(args)
