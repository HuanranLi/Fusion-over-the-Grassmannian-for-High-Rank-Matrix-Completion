import numpy as np
import matplotlib.pyplot as plt
import time

from itertools import permutations
from scipy.optimize import linear_sum_assignment

# def dUU(U_1, U_2, r):
#   u,s,vt = np.linalg.svd(U_1.T @ U_2)
#
#   for i in range(len(s)):
#     if s[i] - 1 > 1e-5:
#       raise Exception('s[',i,'] = ', s[i])
#     elif s[i] > 1:
#       s[i] = 1
#
#   d = sum([np.arccos(s[i])**2 for i in range(r)])
#
#   #print(u,s,vt)
#   assert d >= 0
#   return d


def dUU(U_1, U_2, r):
    # Compute SVD efficiently, assuming you only need the first r singular values
    u, s, vt = np.linalg.svd(U_1.T @ U_2, full_matrices=False)
    s = np.clip(s, a_min=None, a_max=1)  # Ensure that all singular values are <= 1

    # Check if any singular value is significantly greater than 1
    if np.any(s - 1 > 1e-5):
        raise Exception('Singular value greater than 1 encountered')

    # Compute distance using vectorized operations
    d = np.sum(np.arccos(s[:r])**2)

    assert d >= 0
    return d


#  #@title Default title text
# def evaluate(predict, truth, cluster):
#   labels = [i for i in range(cluster)]
#   p = permutations(labels)
#
#   predict = np.array(predict)
#   truth = np.array(truth)
#   assert predict.shape == truth.shape
#
#   err = 1
#   for permuted_label in p:
#     #print("Permutation:", permuted_label)
#     new_predict = np.zeros(len(predict), dtype = int)
#
#     for i in range(len(labels)):
#       new_predict[predict == labels[i]] = int(permuted_label[i])
#
#     err_temp = np.sum(new_predict != truth) / len(predict)
#
#     #print('predict:', new_predict)
#     #print('truth:', truth)
#
#     err = min(err, err_temp)
#     #print("Error Rate:", err_temp)
#
#   return err

#
# import numpy as np

def evaluate(predict, truth, cluster):
    predict = np.array(predict)
    truth = np.array(truth)
    assert predict.shape == truth.shape

    # Create a confusion matrix
    C = np.zeros((cluster, cluster), dtype=int)
    for i in range(len(predict)):
        C[truth[i], predict[i]] += 1

    # Apply the Hungarian algorithm (linear sum assignment)
    row_ind, col_ind = linear_sum_assignment(-C)  # Negate C to find the maximum

    # Calculate the error rate based on the optimal assignment
    error_count = len(predict) - C[row_ind, col_ind].sum()
    err = error_count / len(predict)

    return err
