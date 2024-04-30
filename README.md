# Engineering Synthetic Phosphorylation Signaling Networks in Human Cells

This repository contains the code and data associated with the paper 'Engineering synthetic phosphorylation signaling networks in human cells' authored by X. Yang and colleagues [[bioRxiv](https://www.biorxiv.org/content/10.1101/2023.09.11.557100v2)]. It provides all necessary resources to reproduce the biophysical modeling analysis detailed in the paper.

## Repository Structure

- `data/`: Contains all datasets used in the analysis.
- `src/`: Python scripts for processing and analysis.
- `notebooks/`: Jupyter notebooks to run the analysis and visualize the results.

## Installation

To set up the necessary environment to run the code, you need to install Conda and then create an environment using the provided `environment.yml` file.

### Setting up the Conda Environment

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if it is not already installed.
2. Open a terminal and navigate to the repository directory.
3. Create the Conda environment with:
   ```bash
   conda env create -f environment.yml
   ```
4. Activate the environment:
   ```bash
   conda activate [env_name]
   ```
   Replace [env_name] with the name of the environment as specified at the top of the environment.yml file.

## Usage

After activating the environment, analysis can be run using the Jupyter notebooks in the `notebooks/` folder. Analysis results will be placed in the `results/` folders that can be found for each dataset under the `data/` directory.
   
