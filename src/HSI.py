import scipy.io as sio
import numpy as np
import os
import re

# Function to find the key in the .mat file
def find_mat_key(mat_data):
    keys = [key for key in mat_data if not key.startswith('__')]
    assert len(keys) == 1, "Expected exactly one key in the .mat file"
    return keys[0]

def load_sampled_hsi_data(dataset_idx, num_classes, samples_per_class):
    load_path = '../datasets/HSI/'
    HSI_datasets = ['Indian_pines', 'Pavia', 'PaviaU', 'Salinas', 'SalinasA']
    dataset = HSI_datasets[dataset_idx]

    # Print input parameters
    print(f'Input parameters: dataset = {dataset}, num_classes = {num_classes}, samples_per_class = {samples_per_class}')


    # Check if the '_corrected' file exists
    corrected_file = load_path + dataset + '_corrected.mat'
    if os.path.exists(corrected_file):
        print(f'Loading data from: {corrected_file}')
        X_mat = sio.loadmat(corrected_file)
    else:
        print(f'_corrected file not found. Loading data from: {load_path + dataset + ".mat"}')
        X_mat = sio.loadmat(load_path + dataset + '.mat')

    y_mat = sio.loadmat(load_path + dataset + '_gt.mat')




    # Finding keys
    X_data_key = find_mat_key(X_mat)
    y_data_key = find_mat_key(y_mat)
    # Extracting the data
    X_data = X_mat[X_data_key]
    y_data = y_mat[y_data_key].ravel()
    # Print the shapes before sampling
    print(f'Pre-sampled dataset shape: X: {X_data.shape}, y: {y_data.shape}')

    total_num_classes = len(np.unique(y_data))
    if num_classes > total_num_classes:
        print(f'Number of classes {num_classes} is bigger than total number of classes {total_num_classes}')

    # Reshape X_data
    X_data = X_data.reshape(-1, X_data.shape[2])

    # Identify unique classes
    unique_classes = np.unique(y_data)

    sampled_X = []
    sampled_y = []
    sampled_classes = []
    while len(sampled_classes) < num_classes:
        # Randomly select one class
        selected_class = np.random.choice(unique_classes, 1)[0]

        # Exclude already selected classes
        unique_classes = unique_classes[unique_classes != selected_class]

        # Sampling data for the selected class
        idx = np.where(y_data == selected_class)[0]
        if len(idx) >= samples_per_class:
            selected_idx = np.random.choice(idx, samples_per_class, replace=False)
            sampled_X.append(X_data[selected_idx])
            sampled_y.append(y_data[selected_idx])
            sampled_classes.append(selected_class)

    # Concatenate all samples
    X_sampled = np.concatenate(sampled_X, axis=0).T
    y_sampled = np.concatenate(sampled_y, axis=0)


    unique_sampled_classes = np.unique(y_sampled)
    class_mapping = {original: new for new, original in enumerate(unique_sampled_classes)}
    remapped_y_sampled = np.array([class_mapping[label] for label in y_sampled])


    # Print the shapes after sampling and the sampled classes
    print(f'Post-sampled dataset shape: X: {X_sampled.shape}, y: {remapped_y_sampled.shape}')
    print(f'Classes sampled: {sampled_classes}')

    return X_sampled, remapped_y_sampled




def get_number_of_classes(dataset_idx):
    HSI_datasets = ['Indian_pines', 'Pavia', 'PaviaU', 'Salinas', 'SalinasA']
    load_path = '../datasets/HSI/'
    dataset = HSI_datasets[dataset_idx]

    # Load ground truth data
    y_mat = sio.loadmat(load_path + dataset + '_gt.mat')

    y_data_key = find_mat_key(y_mat)

    # Extracting the ground truth data
    y_data = y_mat[y_data_key].ravel()

    # Count the number of unique classes
    num_classes = len(np.unique(y_data))

    return num_classes



# Example usage
dataset_idx = 0  # For 'Indian_pines'
num_classes = get_number_of_classes(dataset_idx)
print(f'Number of classes in the dataset: {num_classes}')

# Example usage
num_classes = 5
samples_per_class = 50
X_sampled, y_sampled = load_sampled_hsi_data(1, num_classes, samples_per_class)
