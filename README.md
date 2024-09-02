# Grassmannian Fusion for High-Rank Matrix Completion

This repository implements the methods described in the paper "[Fusion Over the Grassmannian for High-Rank Matrix Completion](https://ieeexplore.ieee.org/abstract/document/10619457)", published in IEEE. This project focuses on clustering and completing data matrices with missing entries by leveraging the geometry of the Grassmannian manifold.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Methods](#methods)
- [Datasets](#datasets)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Introduction

High-Rank Matrix Completion (HRMC) is a challenging problem that involves completing and clustering data that lies near a union of subspaces. The method presented in this repository uses the Grassmannian manifold to handle missing data and to ensure local convergence guarantees. The framework does not require prior knowledge of the number of subspaces and is robust to noise, making it well-suited for difficult cases with low sampling rates.

The key contributions of this work include:
- A novel optimization framework over the Grassmannian manifold for HRMC.
- Local convergence guarantees using Riemannian gradient descent.
- Practical techniques for clustering, completion, model selection, and sketching.

For more details, refer to the [published paper](https://ieeexplore.ieee.org/abstract/document/10619457).

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/HuanranLi/Fusion-over-the-Grassmannian-for-High-Rank-Matrix-Completion.git
cd Fusion-over-the-Grassmannian-for-High-Rank-Matrix-Completion
pip install -r requirements.txt
```

## Usage

The main script for running experiments is `main.py`. You can run it with various options depending on the dataset and method you want to use.

### Basic Command

```bash
python main.py --method GF --experiment_name my_experiment --dataset MNIST --num_cluster 3 --samples_per_class 50
```

### Parameters

Here are the primary command-line arguments you can use:

- `--method`: Name of the method to use (`GF`, `ZF_SSC`, `EWZF_SSC`). Default is `GF`.
- `--experiment_name`: Name of the experiment. Default is `test`.
- `--run_name`: Specific run name (optional).
- `--num_runs`: Number of runs to perform. Default is `1`.
- `--step_method`: Method for step size adjustment (default: `Armijo`).
- `--lambda_in`: Regularization parameter (default: `1e-5`).
- `--missing_rate`: Rate of missing data in the dataset (default: `0`).
- `--max_iter`: Maximum number of iterations (default: `50`).
- `--dataset`: Name of the dataset (`Synthetic`, `MNIST`, `Hopkins155`, `HSI`).
- `--step_size`: Step size for optimization (default: `1`).
- `--num_cluster`: Number of clusters (default: `2`).
- `--distance_to_truth`: Flag to calculate distance to the truth (default: `False`).
- `--samples_per_class`: Number of samples per class (default: `50`).
- `--check_acc_per_iter`: Frequency of accuracy checks during clustering (optional).
- `--multiprocessing`: Enable multiprocessing (default: `False`).

### Methods

- **Grassmannian Fusion (GF)**: A method that uses the Grassmannian manifold for clustering with missing data.
- **ZF-SSC**: Zero-Filling Sparse Subspace Clustering.
- **EWZF-SSC**: Entropy-Weighted Zero-Filling Sparse Subspace Clustering.

### Datasets

- **Synthetic Data**: Generate random low-rank matrices with missing entries.
- **MNIST**: Subset of the MNIST digits dataset.
- **Hopkins155**: A dataset for motion segmentation.
- **HSI**: Hyperspectral Image data.

### Results

The results will be logged using `MLFlowLogger` in the specified log directory. Metrics like clustering accuracy and other performance indicators will be saved for each run.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{johnson2024fusion,
  title={Fusion over the Grassmannian for High-Rank Matrix Completion},
  author={Johnson, Jeremy S and Li, Huanran and Pimentel-Alarc{\'o}n, Daniel},
  booktitle={2024 IEEE International Symposium on Information Theory (ISIT)},
  pages={1550--1555},
  year={2024},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
