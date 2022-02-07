#!/usr/bin/env python3

import numpy as np
from mutual_info import compute_cmi, compute_mi
from numpy.linalg import det

# Sample size
N = 20000

# Sample mean
mu = np.zeros(3)

# Covariance matrix
cov_xy = 0.7
cov_xz = 0.5
cov_yz = 0.3
cov = np.array([[1, cov_xy, cov_xz],
                [cov_xy, 1.0, cov_yz],
                [cov_xz, cov_yz, 1]])

samples = np.random.multivariate_normal(mu, cov, size=N)
x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]


# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(x, y, z)
# plt.show()


def cmi_analytic(cov):
    '''Get analytic conditional mutual information for variables x, y given z,
    given a covariance matrix cov.'''
    # Construct minor matrices for x and y
    cov_x = cov[1:, 1:]
    cov_y = cov[[0, 2]][:, [0, 2]]
    return -0.5 * np.log(det(cov) / (det(cov_x) * det(cov_y)))


print("Theory: ", cmi_analytic(cov))
cmi = compute_cmi(x, y, z, 3)
print("cmi: ", cmi)

# I(x;y|z) = I(x;y,z) - I(x;z)
a = compute_mi(x, np.column_stack([y, z]), 3)
b = compute_mi(x, z, 3)
print("I(x;y|z) = I(x;y,z) - I(x;z):", a - b)
