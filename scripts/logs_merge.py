import os
import shutil
import yaml
import argparse

from collections import defaultdict


def find_experiment_folder(base_dir, exp_name):
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        meta_file = os.path.join(folder_path, 'meta.yaml')

        if not os.path.isfile(meta_file):
            continue

        with open(meta_file, 'r') as file:
            meta_data = yaml.safe_load(file)

        if meta_data.get('name') == exp_name:
            return folder, meta_data.get('experiment_id')

    return None, None

def merge_experiments(base_dir, exp_name):
    main_experiment_folder, main_experiment_id = find_experiment_folder(base_dir, exp_name)
    if main_experiment_folder is None or main_experiment_id is None:
        print("No experiment folder found with the given name.")
        return

    main_folder_path = os.path.join(base_dir, main_experiment_folder)

    for folder in os.listdir(base_dir):
        if folder == main_experiment_folder:
            continue

        folder_path = os.path.join(base_dir, folder)
        meta_file = os.path.join(folder_path, 'meta.yaml')

        if not os.path.isfile(meta_file):
            continue

        with open(meta_file, 'r') as file:
            meta_data = yaml.safe_load(file)

        if meta_data.get('name') != exp_name or meta_data.get('experiment_id') == main_experiment_id:
            continue

        # Move sub-folders and then delete the folder
        sub_folders_moved = False
        for sub_folder in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            if not os.path.isdir(sub_folder_path):
                continue

            dest_path = os.path.join(main_folder_path, sub_folder)
            shutil.move(sub_folder_path, dest_path)
            print(f"Moved '{sub_folder_path}' to '{dest_path}'")
            sub_folders_moved = True

            sub_meta_file = os.path.join(dest_path, 'meta.yaml')
            if not os.path.isfile(sub_meta_file):
                continue

            with open(sub_meta_file, 'r') as sub_file:
                sub_meta_data = yaml.safe_load(sub_file)

            sub_meta_data['experiment_id'] = main_experiment_id

            with open(sub_meta_file, 'w') as sub_file:
                yaml.dump(sub_meta_data, sub_file)

        try:
            shutil.rmtree(folder_path)
            print(f"Deleted empty folder '{folder_path}'")
        except OSError as e:
            print(f"Error deleting folder '{folder_path}': {e}")


def find_and_merge_duplicate_experiments(base_dir):
    experiment_folders = defaultdict(list)

    # Iterate through all folders and group by experiment name
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        meta_file = os.path.join(folder_path, 'meta.yaml')

        if not os.path.isfile(meta_file):
            continue

        with open(meta_file, 'r') as file:
            meta_data = yaml.safe_load(file)
            experiment_name = meta_data.get('name')

        if experiment_name:
            experiment_folders[experiment_name].append(folder)

    # Merge experiments that have multiple folders
    for exp_name, folders in experiment_folders.items():
        if len(folders) > 1:
            print(f"Merging experiment '{exp_name}' with {len(folders)} folders")
            merge_experiments(base_dir, exp_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge duplicate experiment subfolders.")

    args = parser.parse_args()

    base_directory = '../logs'
    find_and_merge_duplicate_experiments(base_directory)
