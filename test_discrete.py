#!/usr/bin/env python3

import numpy as np
import argparse
from mutual_info import compute_cmi, compute_mi
from numpy.random import default_rng


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-N', type=int, default=1000, help='num. samples')
    p.add_argument('-k', type=int, default=3, help='num. neighbors')
    p.add_argument('-tests', type=str, nargs='+',
                   choices=['half_discrete', 'mixed', 'discrete', 'bivariate'],
                   default=['half_discrete', 'mixed', 'discrete', 'bivariate'],
                   help='Which tests to perform')
    p.add_argument('-noise_type', type=str,
                   choices=['uniform', 'normal'],
                   help='Which type of noise to add')
    return p.parse_args()


def test_half_discrete(N, n_neighbors, rng, noise):
    """Test case II from "Estimating Mutual Information for Discrete-Continuous
    Mixtures" (Gao et al.)

    """
    m = 5
    x = rng.integers(0, m, size=N)
    y = rng.uniform(x, x+2, size=N)
    z = rng.binomial(3, 0.5, size=N)
    mi_analytic = np.log(m) - (1 - 1/m) * np.log(2)

    cmi = compute_cmi(x, y, z, n_neighbors, noise)
    mi = compute_mi(x, y, n_neighbors, noise)
    return [cmi, mi, mi_analytic]


def test_mixed(N, n_neighbors, rng, noise):
    """Fourth test from "Conditional Mutual Information Estimation for Mixed
    Discrete and Continuous Variables with Nearest Neighbors" (Mesner &
    Shalizi). Similar to test case I "Estimating Mutual Information for
    Discrete-Continuous Mixtures" (Gao et al.)

    """
    mu = np.zeros(2)
    cov = np.array([[1., 0.8], [0.8, 1.0]])
    xy_gauss = rng.multivariate_normal(mu, cov, size=N)

    choices = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    p_discrete = rng.choice(4, p=[0.4, 0.4, 0.1, 0.1], size=N)
    xy_discrete = choices[p_discrete]

    mask = (rng.random(size=N) <= 0.5)
    x = np.where(mask, xy_gauss[:, 0], xy_discrete[:, 0])
    y = np.where(mask, xy_gauss[:, 1], xy_discrete[:, 1])
    z = rng.binomial(3, 0.2, size=N)

    mi_analytic = 0.4 * np.log(2 * 0.4/0.5**2) \
        + 0.1 * np.log(2 * 0.1/0.5**2) \
        + 0.125 * np.log(4/(1 - 0.8**2))

    cmi = compute_cmi(x, y, z, n_neighbors, noise)
    mi = compute_mi(x, y, n_neighbors, noise)
    return [cmi, mi, mi_analytic]


def test_discrete(N, n_neighbors, rng, noise):
    """Third test from "Conditional Mutual Information Estimation for Mixed Discrete
    and Continuous Variables with Nearest Neighbors" (Mesner & Shalizi). Fully
    discrete

    """
    choices = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    p_discrete = rng.choice(4, p=[0.4, 0.4, 0.1, 0.1], size=N)
    xy_discrete = choices[p_discrete]
    x, y = xy_discrete[:, 0], xy_discrete[:, 1]
    z = rng.poisson(2, size=N)

    mi_analytic = 2 * 0.4 * np.log(0.4/0.5**2) + 2 * 0.1 * np.log(0.1/0.5**2)
    cmi = compute_cmi(x, y, z, n_neighbors, noise)
    mi = compute_mi(x, y, n_neighbors, noise)
    return [cmi, mi, mi_analytic]


def test_bivariate(N, n_neighbors, rng, noise):
    """Test with bivariate normal variables"""
    mu = np.zeros(2)
    cov = np.array([[1., 0.8], [0.8, 1.0]])
    xy_gauss = rng.multivariate_normal(mu, cov, size=N)
    x, y = xy_gauss[:, 0], xy_gauss[:, 1]
    z = rng.normal(size=N)

    mi_analytic = -0.5 * np.log(np.linalg.det(cov))
    cmi = compute_cmi(x, y, z, n_neighbors, noise)
    mi = compute_mi(x, y, n_neighbors, noise)
    return [cmi, mi, mi_analytic]


if __name__ == '__main__':
    args = get_args()
    tests = {'half_discrete': test_half_discrete,
             'mixed': test_mixed,
             'discrete': test_discrete,
             'bivariate': test_bivariate}
    rng = default_rng()

    print('{:20} {:9} {:9} {:9} {:9} {:9}'.format(
        '#name', 'cmi', 'mi', 'sol', 'err_cmi', 'err_mi'))

    for t in args.tests:
        f = tests[t]
        cmi, mi, sol = f(args.N, args.k, rng, args.noise_type)
        print(f'{t:20} {cmi:9.2e} {mi:9.2e} {sol:9.2e} {(cmi-sol)/sol:9.2e}'
              + f' {(mi-sol)/sol:9.2e}')
