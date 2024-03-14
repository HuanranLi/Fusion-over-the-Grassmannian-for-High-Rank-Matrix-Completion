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



def distance_to_truth_callback(instance, truth_subspaces, labels):
    distances = np.mean([geodesic(U, truth_subspaces[labels[i]]) for i,U in enumerate(instance.U_array)])
    instance.logger.log_metrics(({'distance_to_truth_mean': distances}), step=instance.iter)

def check_clustering_acc(instance, labels, check_per_iter, num_cluster):
    if instance.iter % check_per_iter != 0:
        return

    d_matrix = instance.distance_matrix()
    similarity_matrix = convert_distance_to_similarity(d_matrix)
    pred_labels, metrics = spectral_clustering(similarity_matrix, num_cluster, labels)

    # print(metrics)
    instance.logger.log_metrics((metrics), step = instance.iter)

    # Assuming mlf_logger is your MLFlowLogger instance
    log_dir = instance.logger.save_dir
    experiment_id = instance.logger.experiment_id  # Or the appropriate method/attribute to get the experiment ID
    run_id = instance.logger.run_id

    # Assuming log_dir is a variable that contains the path to your log directory
    artifact_path = os.path.join(log_dir, experiment_id, run_id, "artifacts")

    # Check if the path exists
    if not os.path.exists(artifact_path):
        print("Artifact path does not exist:", artifact_path)
        raise ValueError()

    saving_file_path = os.path.join(artifact_path, f"U_array_iter{instance.iter}")
    np.savez_compressed(saving_file_path, U_array = instance.U_array, labels = labels)


    print('Successfully save U_array to: ',saving_file_path)
