# fisherA2Z

## Introduction

This repository includes code used for the Fisher Information matrix computation used to assess the impact of photo-z modeling errors on 3x2pt inferences. The code is used and explained in Chapter 4 of Husni Almoubayyed's PhD thesis and an upcoming paper in prep.

## Installation

First, initialize a conda environment by

```
conda create -n 'fisher_env' python=3.8
conda activate fisher_env
```

Clone the repository, and install from source by 

```
pip install -e .
```

Then you can add your conda environment to your jupyterLab by


```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=fisher_env
```


## Code

The majority of the code is in fisher.py, and the Fisher class therein. 

The Fisher class takes a CCL cosmo object and 3 iterables of length 5 each to specify the photo-z error model in terms of biases, standard deviations, and outlier fractions. 


## Examples

To get the fisher matrix for a certain case, it is sufficient to run

```
from fisher import Fisher
f = Fisher(cosmo=ccl_cosmo)
f.process()
```

then the Fisher information matrix will be stored in f.fisher.


## Analysis

The fisher.py contains simple functions that can be ran on the Fisher class to do simple fisher matrix analysis, such as marginalizing over a set of parameters, or plotting 2-dimensional contours

[I have not included the following notebooks yet, they are mostly ready but require some cleaning]

The following notebooks show examples of running the code to get cosmological inferences, compare 2-D confidence contours, and assess the importance of different photo-z error model parameters on cosmological inferences. They are under the Analysis folder.

`fisher.ipynb` shows examples of running the Fisher class and computing simple analysis on the Fisher matrix. It also shows how 2-D Fisher contours compare between different probes and how the contours from our fiducial model compares with another LSST-realistic model in Fig 8 of https://arxiv.org/pdf/2004.07885.pdf.

`Photoz-density-estimation.ipynb` shows how the photo-z outliers were estimated using a KDE from FlexZBoost.

`pz_dists_used.ipynb` shows plots of the fiducial photo-z error models used for the source and lens samples

`lum_dep_IA.ipynb` shows the luminosity-dependent intrinsic alignment implementation

`3x2pt_interpretability.ipynb` notebook shows examples of how the Fisher matrix and a dataset of data-vectors are used to compute the importance of different photo-z error parameters on cosmological inferences, specifically on the bias induced in cosmological parameters when assuming a fixed incorrect photo-z model.

`3x2pt_tree.ipynb` notebook computes the interpretability metrics used in the feature importance computation.