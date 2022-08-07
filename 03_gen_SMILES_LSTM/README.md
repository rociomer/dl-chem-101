# 03_gen_SMILES_LSTM
An example of a SMILES LSTM as a stand-alone package.

In this directory you will find code for:
* SMILES LSTM (full version as stand-alone package)
* A version without any comments, for an exercise in adding comments and docstrings

[TODO write about directory structure here and how things are organized]

# Instructions
## Installing the SMILES LSTM as a package
[TODO]

## Setting up the environment
On mol-gpu-01, you will need to install Miniconda locally to use conda. To do this:
````
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh 
```

To create the environment from scratch using conda, run:
```
conda create -n smiles-lstm-env pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda activate smiles-lstm-env
pip install rdkit
pip install PyTDC
```

To instead create the environment from the provided YAML file, run:
[TODO]

## Downloading the data
For this code, we will be downloading train-test-valid split data from the Therapeutics Data Commons (TDC). We will use ZINC. To  do this, run:
```
python ./data/download-data.py
```

## Running the different generative experiments 
[TODO]
* how to run on cpu locally
* how to run on remote cluster (gpu preferrably)

## Analysis and making figures
[TODO]

