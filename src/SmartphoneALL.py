import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.cluster import DBSCAN
import scipy.io
import collections
from mpl_toolkits.mplot3d import Axes3D

def load_data(sample_size_per_label=100, wanted_words=['min', 'max', 'std', 'mean']):
    f = open('../datasets/dataset_uci/final_y_train.txt', "r")
    f.seek(0)
    lines = f.readlines()
    f.close()
    lines = np.array(lines).astype(int) - 1
    y = lines
    print('Raw y stats:', collections.Counter(y))

    f = open('../datasets/dataset_uci/final_X_train.txt', "r")
    f.seek(0)
    lines = f.readlines()
    f.close()
    # Handling comma-separated values
    X = np.array([np.array(i.strip().split(',')).astype(float) for i in lines]).T
    print('Raw X shape:', X.shape)
    Xs = [X[:, y == i] for i in set(y)]

    f = open('../datasets/dataset_uci/features.txt')
    f.seek(0)
    lines = f.readlines()
    f.close()
    need_row = []

    for i, row in enumerate(lines):
        for word in wanted_words:
            if word in row:
                need_row.append(i)
                break

    X_cutted = X[need_row, :]

    y_remap = y.copy()
    y_remap[y == 1] = 0
    y_remap[y == 2] = 0
    y_remap[y == 3] = 1
    y_remap[y == 4] = 1
    y_remap[y == 5] = 1

    ys = [np.where(y_remap == i)[0] for i in [0, 1]]

    ys_sampled = [np.random.choice(ys_i, size=sample_size_per_label, replace=False) for ys_i in ys]
    X_final = np.concatenate((X_cutted[:, ys_sampled[0]], X_cutted[:, ys_sampled[1]]), axis=1)
    y_final = np.concatenate(([0] * sample_size_per_label, [1] * sample_size_per_label))

    print('X_final shape:', X_final.shape)
    print('y_final shape:', y_final.shape)

    return X_final, y_final

def main():
    # data can be downloaded at: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/
    X, y = load_data(sample_size_per_label=50, wanted_words=['min', 'max', 'std', 'mean'])

if __name__ == '__main__':
    main()
