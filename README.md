# dl-chem-101

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

We recommend two ways to use this repository
1. Reattempting tasks 

Students should consider attempting these tasks from scratch before viewing our solutions 
(see Problem Prompts)

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
### 02_prop_pred_GNN
### 03_gen_smiles_LSTM
### 04_gen_smiles_VAE
