import numpy as np
import scipy.io as sio
import os


def process_mat_data(file_path):
    """
    Processes the data from a .mat file, extracts and reshapes the 'x' array,
    and reads the 's' array.

    Parameters:
    file_path (str): Path to the .mat file.

    Returns:
    np.ndarray: The reshaped 'x' array.
    np.ndarray: The 's' array.
    """
    # Load the MATLAB file
    data = sio.loadmat(file_path)

    # Reading 'x' from the data
    x = data['x']
    # print("Original shape of x:", x.shape)

    # Extracting the first two channels of 'x'
    x_first_two_channels = x[:2, :, :]
    # print("Shape after extracting first two channels of x:", x_first_two_channels.shape)

    # Swapping the dimensions
    x_first_two_channels_swapped = np.transpose(x_first_two_channels, (1, 2, 0))
    # print("Shape after swapping dimensions:", x_first_two_channels_swapped.shape)

    # Reshaping to flatten the last two dimensions
    x_flattened_swapped = x_first_two_channels_swapped.reshape(x_first_two_channels_swapped.shape[0], -1).T

    # Reading 's' from the data
    s = data['s'].flatten() - 1

    print("Final shape after reshaping:", x_flattened_swapped.shape)
    print("Shape of s:", s.shape)
    print("Number of Classes:", len(set(s)))

    if not (max(s) < len(set(s)) and min(s) >=0):
        print(s)
        assert False



    return x_flattened_swapped, s


def process_hopkins_sequence(hopkins_path, index):
    # Get all folder names under hopkins_path
    folders = sorted([f for f in os.listdir(hopkins_path) if os.path.isdir(os.path.join(hopkins_path, f))])
    print('Total Number of Hpkins Trails: ', len(folders))
    print('Target Number: ', index)

    # Select the folder based on the provided index
    selected_folder = folders[index]
    folder_path = os.path.join(hopkins_path, selected_folder)

    # Assuming there's one main .mat file per folder, find it
    for file in os.listdir(folder_path):
        if file.endswith('.mat') and not file.startswith('.'):
            file_path = os.path.join(folder_path, file)
            break

    print('Reading: ', file_path)
    # Load the MATLAB file
    data, label = process_mat_data(file_path)


    return np.array(data), np.array(label)
