# This code is heavily inspired by sklearn/feature_selection/_mutual_info.py,
# which was written by Nikolay Mayorov <n59_ru@hotmail.com> under the 3-clause
# BSD license.
#
# Author: Jannis Teunissen <jannis.teunissen@cwi.nl>

import numpy as np
from numpy.random import default_rng
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


def ensure_2d(x, rng=default_rng()):
    """Ensure ndarray is 2d and add a small amount of noise to avoid duplicate
    samples

    :param x: ndarray, shape (n_samples,) or (n_samples, n_features)
    :returns: float ndarray, shape (n_samples, n_features)

    """
    # Ensure 2D and dtype float
    if x.ndim == 1:
        x = x.copy().astype(float)[:, np.newaxis]
    elif x.ndim == 2:
        x = x.copy().astype(float)
    else:
        raise ValueError('x.ndim not equal to 1 or 2')

    # Add noise to ensure samples are unique, see Kraskov et. al.
    # means = np.maximum(1, np.mean(np.abs(x), axis=0))
    # Use 'cheap' uniform random numbers
    # x += 1e-10 * means * (rng.random(x.shape) - 0.5)
    return x


def compute_mi(x, y, n_neighbors=3):
    """Compute mutual information between two continuous variables.

    :param x: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param y: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param n_neighbors: Number of nearest neighbors
    :returns: non-negative estimate of mutual information
    """
    if len(x) != len(y):
        raise ValueError('x and y should have the same length')

    x = ensure_2d(x)
    y = ensure_2d(y)
    xy = np.hstack((x, y))
    n_samples = len(x)
    k = np.full(n_samples, n_neighbors)

    radius = get_radius_kneighbors(xy, n_neighbors)

    # Where radius is 0, determine multiplicity
    mask = (radius == 0)
    vals, ix, counts = np.unique(xy[mask], axis=0, return_inverse=True,
                                 return_counts=True)
    k[mask] = counts[ix] - 1

    nx = num_points_within_radius(x, radius)
    ny = num_points_within_radius(y, radius)

    mi = max(0, digamma(n_samples) + np.mean(digamma(k))
             - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))
    return mi


def compute_cmi(x, y, z, n_neighbors=3):
    """Compute conditional mutual information I(x;y|z)

    :param x: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param y: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param z: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param n_neighbors: Number of nearest neighbors
    :returns: non-negative estimate of conditional mutual information

    """
    x = ensure_2d(x)
    y = ensure_2d(y)
    z = ensure_2d(z)

    radius = get_radius_kneighbors(np.hstack((x, y, z)), n_neighbors)
    nxz = num_points_within_radius(np.hstack((x, z)), radius)
    nyz = num_points_within_radius(np.hstack((y, z)), radius)
    nz = num_points_within_radius(z, radius)

    cmi = max(0, digamma(n_neighbors) - np.mean(digamma(nxz + 1))
              - np.mean(digamma(nyz + 1)) + np.mean(digamma(nz + 1)))
    return cmi


# def compute_mi_discrete(x, yd, n_neighbors=3):
#     """Compute mutual information between continuous x and discrete yd.

#     :param x: real ndarray, shape (n_samples,) or (n_samples, n_features)
#     :param yd: discrete ndarray, shape (n_samples,)
#     :param n_neighbors: Number of nearest neighbors
#     :returns: non-negative estimate of mutual information

#     """
#     x = ensure_2d(x)

#     labels, ix, counts = np.unique(yd, return_inverse=True, return_counts=True)

#     # Count for each point in yd
#     yd_counts = counts[ix]

#     # Ignore labels with fewer than n_neighbors + 1 points
#     mask = (yd_counts > n_neighbors)
#     x = x[mask]
#     yd = yd[mask]
#     yd_counts = yd_counts[mask]

#     label_mask = (counts > n_neighbors)
#     labels = labels[label_mask]
#     counts = counts[label_mask]

#     n_samples = counts.sum()
#     radius = np.empty(n_samples)

#     for label, count in zip(labels, counts):
#         mask = (yd == label)
#         radius[mask] = get_radius_kneighbors(x[mask], n_neighbors)

#     nx = num_points_within_radius(x, radius)

#     mi = max(0, digamma(n_samples) + digamma(n_neighbors)
#              - np.mean(digamma(yd_counts)) - np.mean(digamma(nx + 1)))
#     return mi
