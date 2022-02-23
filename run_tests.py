#!/usr/bin/env python3

import numpy as np
import argparse
from mutual_info import compute_cmi, compute_mi
from numpy.random import default_rng
from numpy.linalg import det


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-N', type=int, default=1000, help='num. samples')
    p.add_argument('-k', type=int, default=3, help='num. neighbors')
    p.add_argument('-n_runs', type=int, default=10,
                   help='Number of runs of the test')
    p.add_argument('-tests', type=str, nargs='+',
                   choices=['half_discrete', 'mixed', 'discrete', 'bivariate',
                            'trivariate'],
                   default=['half_discrete', 'mixed', 'discrete', 'bivariate',
                            'trivariate'],
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
    cmi_analytic = np.log(m) - (1 - 1/m) * np.log(2)

    cmi = compute_cmi(x, y, z, n_neighbors, noise)
    mi = compute_mi(x, y, n_neighbors, noise)
    return [cmi, mi, cmi_analytic]


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

    cmi_analytic = 0.4 * np.log(2 * 0.4/0.5**2) \
        + 0.1 * np.log(2 * 0.1/0.5**2) \
        + 0.125 * np.log(4/(1 - 0.8**2))

    cmi = compute_cmi(x, y, z, n_neighbors, noise)
    mi = compute_mi(x, y, n_neighbors, noise)
    return [cmi, mi, cmi_analytic]


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

    cmi_analytic = 2 * 0.4 * np.log(0.4/0.5**2) + 2 * 0.1 * np.log(0.1/0.5**2)
    cmi = compute_cmi(x, y, z, n_neighbors, noise)
    mi = compute_mi(x, y, n_neighbors, noise)
    return [cmi, mi, cmi_analytic]


def test_bivariate(N, n_neighbors, rng, noise):
    """Test with bivariate normal variables"""
    mu = np.zeros(2)
    cov = np.array([[1., 0.8], [0.8, 1.0]])
    xy_gauss = rng.multivariate_normal(mu, cov, size=N)
    x, y = xy_gauss[:, 0], xy_gauss[:, 1]
    z = rng.normal(size=N)

    cmi_analytic = -0.5 * np.log(det(cov))
    cmi = compute_cmi(x, y, z, n_neighbors, noise)
    mi = compute_mi(x, y, n_neighbors, noise)
    return [cmi, mi, cmi_analytic]


def test_trivariate(N, n_neighbors, rng, noise):
    """Test with 'trivariate' normal variables x, y, z"""
    mu = np.zeros(3)

    # Covariance matrix
    cov_xy = 0.7
    cov_xz = 0.5
    cov_yz = 0.3
    cov = np.array([[1, cov_xy, cov_xz],
                    [cov_xy, 1.0, cov_yz],
                    [cov_xz, cov_yz, 1]])

    samples = rng.multivariate_normal(mu, cov, size=N)
    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]

    # Construct minor matrices for x and y
    cov_x = cov[1:, 1:]
    cov_y = cov[[0, 2]][:, [0, 2]]
    cmi_analytic = -0.5 * np.log(det(cov) / (det(cov_x) * det(cov_y)))

    cmi = compute_cmi(x, y, z, n_neighbors, noise)

    # Estimate via I(x;y|z) = I(x;y,z) - I(x;z)
    mi = compute_mi(x, np.column_stack([y, z]), n_neighbors, noise) - \
        compute_mi(x, z, n_neighbors, noise)
    return [cmi, mi, cmi_analytic]


if __name__ == '__main__':
    args = get_args()
    tests = {'half_discrete': test_half_discrete,
             'mixed': test_mixed,
             'discrete': test_discrete,
             'bivariate': test_bivariate,
             'trivariate': test_trivariate}
    rng = default_rng()

    print('{:4} {:7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:20}'.format(
        'k', 'N', 'cmi', 'mi', 'sol', 'err_cmi', 'err_mi',
        'std_cmi', 'std_mi', '#test name'))

    for t in args.tests:
        f = tests[t]

        results = np.zeros((args.n_runs, 3))

        for i in range(args.n_runs):
            results[i] = f(args.N, args.k, rng, args.noise_type)

        cmi, mi, sol = np.mean(results, axis=0)
        std_cmi, std_mi = np.std(results[:, 0:2], axis=0, ddof=1)
        print(f'{args.k:<4} {args.N:<7} {cmi:9.2e} {mi:9.2e} {sol:9.2e}'
              + f' {cmi-sol:9.2e} {mi-sol:9.2e} {std_cmi:9.2e}'
              + f' {std_mi:9.2e} #{t}')
