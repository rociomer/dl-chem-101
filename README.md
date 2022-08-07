# dl-chem-101

A repository for various introductory tutorials on deep learning for chemistry [TODO modify this text and description later].

* [01_prop_pred_FFN](./01_prop_pred_FFN/)
* [02_prop_pred_GNN](./02_prop_pred_GNN/)
* [03_gen_SMILES_LSTM](./03_gen_SMILES_LSTM/)
* [04_gen_SMILES_VAE](./04_gen_SMILES_VAE/)

## Introduction

There is often a gap between computer science and chemistry coursework and
research. Most classes now support Jupyter Notebooks or Google Colab
enviornments that have simple install, setup, and often require running code
with a single setup. While very useful and didactic, we find there is a missing
course explaining how students can structure repositories for new research projects 
that enable them to organize new experiments, try different model settings, and
move quickly.

This repository is an opinionated attempt to show several ways to structure
these repositories for basic tasks we expect any researcher at the intersection of
machine learning and chemistry to implement. Specifically:

1. Molecular property prediction with feed forward networks   
2. Molecular property prediction with graph neural networks  
3. Molecular generation with a SMILES LSTM  
4. Molecular generation with a VAE   

### How should I use this? 

We recommend two ways to use this repository:
1. Reattempting tasks 

Students should consider attempting these tasks from scratch before viewing our solutions 
(see Problem Prompts).

2. Adding documentation

We recognize that attempting these may be too time consumign for shorter
onboarding periods. As an alternative, we provide versions of the code with _no
documentation_ (TODO: Add). As a useful exercise, we recommend running the code and then
adding documentation (TODO: Add recommendation for documentation, like adding
tensor shapes, types, etc.). 

### Learning outcomes

After either reattempting these tasks or adding documentation, you can be
expected to understand:
1. How to structure an ML for chemistry repository
2. How to launch experiments for various model parameters and configurations
3. How to separate analysis from model predictions

## Problem Prompts

For those interested in attempting this on their own before viewing our
solutions and structures, we provide the following guiding prompts and
references.

### 01_prop_pred_FFN
In this repository, we will use a feed-forward neural network (FFN) to predict a molecular property relevant to drug discovery, Caco-2 cell permeability, from [molecular fingerprints](https://doi.org/10.1021/ci100050t) (originally demonstrated in [Wang *et al.* (2016)](https://doi.org/10.1021/acs.jcim.5b00642)). We use data available for download via the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al) (original paper introducing the TDC from [Huang *et al.* (2021)](https://arxiv.org/abs/2102.09548)).


### 02_prop_pred_GNN

[Gilmer *et al.* (2017)](https://proceedings.mlr.press/v70/gilmer17a.html)
[Yang *et al.* (2019)](https://doi.org/10.1021/acs.jcim.9b00237)
[Duvenaud *et al.* (2015)](https://proceedings.neurips.cc/paper/2015/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html)
MoleculeNet [Wu *et al.* (2017)](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a#cit63) 

### 03_gen_smiles_LSTM
In this repository, we will go through the process of training a SMILES long short-term memory (LSTM) network for molecular design tasks. At a high level, the way the model works is that it sees examples of valid molecular SMILES strings, and thus learns to generate new ones by sampling tokens from conditional probability distributions, similar to natural language processing tools which predict the best way to finish a sentence having seen the first few words.

While the SMILES LSTM is widely-used in deep learning for molecular design tasks, one of the first papers to report a recurrent neural network (RNN) for molecular generation was [Segler *et al.* (2018)](https://doi.org/10.1021/acscentsci.7b00512). Another good early paper on a generative SMILES LSTM is [Bjerrum, E. J. (2018)](https://arxiv.org/abs/1703.07076).

Code for this example was adapted from the SMILES LSTM implementation in the Molecular AI [REINVENT](https://github.com/MolecularAI/Reinvent) repository and structured as a stand-alone package.

In this example, we train on a 50K subset of SMILES strings from the ZINC dataset [available from the TDC](https://tdcommons.ai/generation_tasks/molgen/). We go through the process of downloading and splitting the data, then provide two examples for training the model: one on a local GPU, and one on a supercomputing cluster using a job scheduler (LLsub).

### 04_gen_smiles_VAE


## References
1. Segler *et al.*, *ACS Cent. Sci.*, **2018**, 4, 1, *120–131*.