import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import Lasso

def solve_l1_optimization(data, lambda_val=1.0):
    """
    Solve the L1-optimization problem for sparse representation.

    :param data: Data matrix (each column is a data point)
    :param lambda_val: Regularization parameter for L1-norm
    :return: Sparse coefficient matrix
    """
    num_points = data.shape[1]
    coef_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        # Creating the optimization problem
        A = np.concatenate([data[:, :i], data[:, i+1:]], axis=1)
        b = data[:, i]

        # L1-optimization (Lasso)
        clf = Lasso(alpha=lambda_val, fit_intercept=False)
        clf.fit(A, b)
        x = clf.coef_

        # Inserting the solution into the coefficient matrix
        coef_matrix[i, :i] = x[:i]
        coef_matrix[i, i+1:] = x[i:]

    return coef_matrix

def sparse_subspace_clustering(data, lambda_val=1.0):
    """
    Perform Sparse Subspace Clustering (SSC).

    :param data: Data matrix (each column is a data point)
    :param num_clusters: Number of clusters to form
    :param lambda_val: Regularization parameter for L1-norm in L1-optimization
    :return: Cluster labels for each data point
    """
    # Step 1: Sparse Representation
    coef_matrix = solve_l1_optimization(data, lambda_val)

    # Step 2: Construct Affinity Matrix
    affinity_matrix = 0.5 * (np.abs(coef_matrix) + np.abs(coef_matrix.T))


    return affinity_matrix


def zf_ssc(data, observed_mask, lambda_val=1e-1):
    """
    Perform Zero-Filled Sparse Subspace Clustering (ZF-SSC).

    :param data: Data matrix (each column is a data point)
    :param observed_mask: Boolean matrix indicating observed data points
    :param lambda_val: Regularization parameter for L1-norm in L1-optimization
    :return: Affinity matrix
    """
    # Step 1: Apply observed mask - set unobserved values to zero
    n, m = data.shape
    mask = create_mask_from_indices((n, m), observed_mask)
    zero_filled_data = np.where(mask, data, 0)

    # Step 2: Sparse Representation
    coef_matrix = solve_l1_optimization(zero_filled_data, lambda_val)

    # Step 3: Construct Affinity Matrix
    affinity_matrix = 0.5 * (np.abs(coef_matrix) + np.abs(coef_matrix.T))

    return affinity_matrix


def create_mask_from_indices(shape, indices):
    """
    Create a boolean mask from a list of indices.

    :param shape: The shape of the mask (should match the data shape).
    :param indices: List of linear indices to be marked as True.
    :return: A boolean mask.
    """
    mask = np.zeros(shape, dtype=bool)
    for idx in indices:
        row, col = divmod(idx, shape[1])
        mask[row, col] = True
    return mask

# Assuming 'data' is your data matrix and 'observed_mask' is a list of indices
# n, m = data.shape
# mask = create_mask_from_indices((n, m), observed_mask)
# zero_filled_data = np.where(mask, data, 0)
