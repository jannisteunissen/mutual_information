# This code is heavily inspired by sklearn/feature_selection/_mutual_info.py,
# which was written by Nikolay Mayorov <n59_ru@hotmail.com> under the 3-clause
# BSD license.
#
# Author: Jannis Teunissen <jannis.teunissen@cwi.nl>

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree


def get_radius_kneighbors(x, n_neighbors):
    """Determine smallest radius around x containing n_neighbors neighbors

    :param x: ndarray, shape (n_samples, n_dim)
    :param n_neighbors: number of neighbors
    :returns: radius, shape (n_samples,)

    """
    # Use KDTree for simplicity (sometimes a ball tree could be faster)
    kd = KDTree(x, metric="chebyshev")

    # Results include point itself, therefore n_neighbors+1
    neigh_dist = kd.query(x, k=n_neighbors+1)[0]

    # Take radius slightly larger than distance to last neighbor
    radius = np.nextafter(neigh_dist[:, -1], 0)
    return radius


def num_points_within_radius(x, radius):
    """For each point, determine the number of other points within a given radius

    :param x: ndarray, shape (n_samples, n_dim)
    :param radius: radius, shape (n_samples,)
    :returns: number of points within radius

    """
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    return np.array(nx) - 1.0


def ensure_2d_add_noise(x):
    """Ensure ndarray is 2d and add small noise"""
    if x.ndim == 1:
        y = x.copy()[:, np.newaxis]
        return y
    else:
        return x


def compute_mi(x, y, n_neighbors):
    """Compute mutual information between two continuous variables.

    Parameters
    ----------
    x, y : ndarray, shape (n_samples,) or (n_samples, n_features)
        Samples of two continuous random variables

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information (non-negative)

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    if len(x) != len(y):
        raise ValueError('x and y should have the same length')

    x = ensure_2d_add_noise(x)
    y = ensure_2d_add_noise(y)
    xy = np.hstack((x, y))
    n_samples = len(x)

    radius = get_radius_kneighbors(xy, n_neighbors)
    nx = num_points_within_radius(x, radius)
    ny = num_points_within_radius(y, radius)

    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )

    return max(0, mi)


def compute_cmi(x, y, z, n_neighbors):
    """Compute conditional mutual information I(x;y|z)

    Parameters
    ----------
    x, y, z : ndarray, shape (n_samples,)
        Samples of three continuous random variables, must have an identical
        shape.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    cmi : float
        Estimated conditional mutual information.

    """
    x = ensure_2d_add_noise(x)
    y = ensure_2d_add_noise(y)
    z = ensure_2d_add_noise(z)
    xz = np.hstack((x, z))
    yz = np.hstack((y, z))
    xyz = np.hstack((x, y, z))

    radius = get_radius_kneighbors(xyz, n_neighbors)
    nxz = num_points_within_radius(xz, radius)
    nyz = num_points_within_radius(yz, radius)
    nz = num_points_within_radius(z, radius)

    cmi = (
        digamma(n_neighbors)
        - np.mean(digamma(nxz + 1))
        - np.mean(digamma(nyz + 1))
        + np.mean(digamma(nz + 1))
    )

    return cmi
