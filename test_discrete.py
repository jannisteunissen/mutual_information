#!/usr/bin/env python3

import numpy as np
from mutual_info import compute_cmi, compute_mi

# Sample size
N = 10*1000
n_neighbors = 3

# Test case II from "Estimating Mutual Information for Discrete-Continuous
# Mixtures" (Gao et al.)
m = 5
x = np.random.randint(0, m, size=N)
y = np.random.uniform(x, x+2, size=N)
z = np.random.binomial(3, 0.5, size=N)
mi_analytic = np.log(m) - (1 - 1/m) * np.log(2)

print("Theory: ", mi_analytic)
print("cmi: ", compute_cmi(x, y, z, n_neighbors))
print("mi: ", compute_mi(x, y, n_neighbors))

# Fourth test from "Conditional Mutual Information Estimation for Mixed Discrete
# and Continuous Variables with Nearest Neighbors" (Mesner & Shalizi). Similar
# to test case I "Estimating Mutual Information for Discrete-Continuous
# Mixtures" (Gao et al.)
mu = np.zeros(2)
cov = np.array([[1., 0.8], [0.8, 1.0]])
xy_gauss = np.random.multivariate_normal(mu, cov, size=N)

choices = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
p_discrete = np.random.choice(4, p=[0.4, 0.4, 0.1, 0.1], size=N)
xy_discrete = choices[p_discrete]

mask = (np.random.random(size=N) <= 0.5)
x = np.where(mask, xy_gauss[:, 0], xy_discrete[:, 0])
y = np.where(mask, xy_gauss[:, 1], xy_discrete[:, 1])

mi_analytic = 0.4 * np.log(2 * 0.4/0.5**2) \
    + 0.1 * np.log(2 * 0.1/0.5**2) \
    + 0.125 * np.log(4/(1 - 0.8**2))
print("Theory: ", mi_analytic)
print("mi: ", compute_mi(x, y, n_neighbors))

# Third test from "Conditional Mutual Information Estimation for Mixed Discrete
# and Continuous Variables with Nearest Neighbors" (Mesner & Shalizi).
choices = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
p_discrete = np.random.choice(4, p=[0.4, 0.4, 0.1, 0.1], size=N)
xy_discrete = choices[p_discrete]
x, y = xy_discrete[:, 0], xy_discrete[:, 1]

mi_analytic = 2 * 0.4 * np.log(0.4/0.5**2) + 2 * 0.1 * np.log(0.1/0.5**2)
print("Theory: ", mi_analytic)
print("mi: ", compute_mi(x, y, n_neighbors))
