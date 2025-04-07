# porosity_reconstruction
The objective of this assignment is to develop an efficient and robust method or algorithm to reconstruct the distribution of pores within a three-dimensional space, based on a given density factor. Awarded best assignment among all candidates.

The dataset provides numerical representations of pore sizes and their spatial distribution within a unit cube (ranging from 0 to 1 in all dimensions). Each simulation instance is associated with a density factor between 0 and 1, indicating the proportion of the volume occupied by spherical pores. Higher values correspond to greater pore occupancy. Each input file contains a NumPy array representing a flattened 3D grid with a resolution of 30 grid points per axis. The data structure consists of rows, where each row includes four values: the x, y, and z coordinates, along with an occupancy flag (0 or 1) indicating whether the corresponding point is inside a pore.

Evaluation Criteria: 
- Accuracy of pore reconstruction
- Computational efficiency and speed
- Innovativeness and robustness of the method
- Clarity and completeness of documentation
- Quality and comprehensiveness of the accompanying report.

## Approach
This project explores a novel loss function for the Conditional Variational Autoencoder to tackle with the sparsity problem of the reconstructed grid, as well as incorporating structural properties of the pore distribution. The model has been developped using **Pytorch** and **Pytorch-Lightning**.

## Prerequisites
**Python Environment Setup**

The python environement for this project is managed through [Poetry](https://python-poetry.org/). Poetry can be installed with the following command:
```sh
pip install poetry
poetry --version
```

Install the Poetry env with:
```sh
poetry config virtualenvs.in-project true
poetry install
```

All dependecies are explicitely listed in the file **pyproject.toml**.

**Data Storage**

Data should be stored in a **data** folder located in the main directory.

**Computer Setup**

The trainings of the models were conduted on a Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz with 32GB of RAM and a NVIDIA Quadro T2000 GPU. The training of the model takes less than *15 minutes*, and once trained, the inference time for the full dataset takes *3 seconds*. Due to their size, the trained weights cannot be shared in this repository.

## Structure of the Repository

- ***models***: contains the script building the models classes.
- ***utils***: contains scripts for data processing, data visualisation, and building a Pytorch DataLoader.
- ***notebooks***: contains notebooks for data exploration, model training, and plots reproduction.
- ***reports***: contains the pdfs for the assignement and the final report.

## Running Instructions

Data exploration and its corresponding plots can be reproduced when running the notebook ***notebooks/00_exploration.ipynb***

The training of the Conditional Variational Autoencoder as well as the display of its results can be reproduced when running the notebook ***notebooks/01_vae_training.ipynb***. When training, the logs will be sotred in a directory called **lightning_logs** located in the parent directory.



