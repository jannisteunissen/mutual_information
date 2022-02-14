# Mutual information

Python routines to compute (conditional) mutual information

## Overview

Both [mutual information](https://en.wikipedia.org/wiki/Mutual_information) and
[conditional mutual
information](https://en.wikipedia.org/wiki/Conditional_mutual_information) can
be computed with this module, using a nearest-neighbor algorithm.

## Requirements

* numpy
* scipy (only scipy.special.digamma)
* scikit-learn (only sklearn.neighbors.KDTree)

## Usage

Compute mutual information between x and y, which can be 1d or 2d arrays:

    mi = compute_mi(x, y, n_neighbors=3, noise_type=None)

Similarly, to compute conditional mutual information:

    cmi = compute_cmi(x, y, n_neighbors=3, noise_type=None)

## Method

A nearest neighbor approach is used, for which theoretical background is provided in the following papers and references therein:
* [Estimating mutual information](https://doi.org/10.1103/PhysRevE.69.066138), by Kraskov et al.
* [Estimating Mutual Information for Discrete-Continuous Mixtures](https://proceedings.neurips.cc/paper/2017/file/ef72d53990bc4805684c9b61fa64a102-Paper.pdf), by Gao et al.
* [Conditional Mutual Information Estimation for Mixed Discrete and Continuous Variables with Nearest Neighbors](http://arxiv.org/abs/1912.03387), by Mesner and Shalizi

The basic idea is to first determine the [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance) $\rho_i$ to the k-th nearest neighbor of each sample. For each sample, the following quantity can then be computed:

$\xi_i = \phi(k) + \phi(N) - \phi(n_x + 1) - \phi(n_y + 1)$

where $N$ is the total number of samples, and $n_x$ and $n_y$ are the number of samples within a distance $\rho_i$ if only the $x$ and $y$ coordinates are considered, respectively. The mutual information between is then estimated as the mean value of $\xi_i$

$mi(x, y) = <\xi_i>$

Conditional mutual information can be estimated in two ways, either by using the identity

$I(x;y|z) = I(x;y,z) - I(x;z)$

or by using a slightly modified estimator. The radius $\rho_i$ is first computed on the $x, y, z$ data, and then $\xi_i$ for each sample point is given by

$\xi_i = \phi(k) + \phi(n_z+1) - \phi(n_{xz} + 1) - \phi(n_{yz} + 1)$

## Implementation

The implementation is inspired by and based on the mutual information methods
available in [scikit-learn](scikit-learn.org/), which where implemented by
Nikolay Mayorov. The nearest neighbor searches are performed using the k-d tree
implementation provided by scikit-learn.

## Acknowledgements

This project has received funding from the European Union’s Horizon 2020 Research and Innovation programme under grant agreement No 776262 [AIDA](http://aida-space.eu/).

## Related projects

* https://github.com/sudiptodip15/CCMI
* https://github.com/majianthu/pycopent
* https://github.com/JuliaDynamics/TransferEntropy.jl
* https://github.com/omesner/knncmi
* https://github.com/yandex/CMICOT
