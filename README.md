# porosity_reconstruction
The objective of this assignment is to develop an efficient and robust method or algorithm to reconstruct the distribution of pores within a three-dimensional space, based on a given density factor.

The dataset provides numerical representations of pore sizes and their spatial distribution within a unit cube (ranging from 0 to 1 in all dimensions). Each simulation instance is associated with a density factor between 0 and 1, indicating the proportion of the volume occupied by spherical pores. Higher values correspond to greater pore occupancy. Each input file contains a NumPy array representing a flattened 3D grid with a resolution of 30 grid points per axis. The data structure consists of rows, where each row includes four values: the x, y, and z coordinates, along with an occupancy flag (0 or 1) indicating whether the corresponding point is inside a pore.

Evaluation Criteria: 
- Accuracy of pore reconstruction
- Computational efficiency and speed
- Innovativeness and robustness of the method
- Clarity and completeness of documentation
- Quality and comprehensiveness of the accompanying report.

## Approach
This project explores a novel loss function for the Conditional Variational Autoencoder to tackle with the sparsity problem of the reconstructed grid, as well as incorporating structural properties of the pore distribution.

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
Data should be stored in a **data** folder in the main directory.

**Computer Setup**
The trainings of the models were conduted on a Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz with 32GB of RAM and a NVIDIA Quadro T2000 GPU.
