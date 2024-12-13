# ARTEMIS
 
> **A**lgorithm for **R**are-variant **T**rait **E**valuation using **M**odel-based **I**nference and **S**election


**ARTEMIS** is Python-based tool for performing Bayesian variable selection in a classification model for observations with binary features, applied primarily to rare-variant gene-level collapsing data from UK Biobank.
Demonstrated on a synthetic binary-valued dataset.


## Installation
To install this python package, first download (clone) the git repo.
Creating a new conda environment is recommended, commands to do this are listed below. 
```
conda create -n ARTEMIS python=3.10
conda activate ARTEMIS
pip install -e .
```
**Note** that one must be in the git root directory to install. 


## Examples
A typical workflow for performing variational inference is as follows:

```
from artemis import artemis

# Initialise the variational inference module
vb = artemis.VB(
    X, # binary genotype matrix
    gene_names, # name of genes corresponding to each column of the genotype matrix
    y # phenotype labels (should be a binary vector of length matching the number of rows of X)
)

# Run variational inference (n_rep=number of iterations)
vb.run(n_rep=3)
```

See the `examples.ipynb` notebook for progressively more sophisticated examples. 
Note that to use the pipeline (last example in the notebook), one must first generate synthetic data using `generate_synthetic_pipeline_data.ipynb`.

This is not required to get started.

## Citation
```
```