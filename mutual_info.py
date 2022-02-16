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


def preprocess_data(x):
    """Preprocess data. Ensure x is 2d ndarray, and scale so that the mean absolute
    amplitude of each column is one.

    :param x: ndarray, shape (n_samples,) or (n_samples, n_features)
    :returns: float ndarray, shape (n_samples, n_features)

    """
    x = np.array(x, dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError(f'x.ndim = {x.ndim}, should be 1 or 2')

    # Estimate mean absolute amplitude per column
    means = np.maximum(1e-100, np.mean(np.abs(x), axis=0))

    # Scale so that mean absolute amplitude is one
    x = (1/means) * x

    return x


def add_noise(x, rng, noise_type='uniform', amplitude=1e-10):
    """Add noise so that samples are probably unique, and convert to float64"""

    if noise_type == 'uniform':
        x += amplitude * (rng.random(x.shape) - 0.5)
    elif noise_type == 'normal':
        x += amplitude * rng.normal(size=x.shape)
    else:
        raise ValueError('Invalid noise type')

    return x


def compute_mi(x, y, n_neighbors=3, noise_type=None):
    """Compute mutual information between two continuous variables.

    :param x: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param y: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param n_neighbors: Number of nearest neighbors
    :param noise_type: add noise of given type (uniform, normal)
    :returns: non-negative estimate of mutual information

    """
    n_samples = len(x)
    x, y = [preprocess_data(t) for t in [x, y]]

    if noise_type:
        rng = default_rng()
        x, y = [add_noise(t, rng, noise_type) for t in [x, y]]

    xy = np.hstack((x, y))
    k = np.full(n_samples, n_neighbors)
    radius = get_radius_kneighbors(xy, n_neighbors)

    if noise_type is None:
        # Where radius is 0, determine multiplicity
        mask = (radius == 0)
        if mask.sum() > 0:
            vals, ix, counts = np.unique(xy[mask], axis=0, return_inverse=True,
                                         return_counts=True)
            k[mask] = counts[ix] - 1

    nx = num_points_within_radius(x, radius)
    ny = num_points_within_radius(y, radius)

    mi = max(0, digamma(n_samples) + np.mean(digamma(k))
             - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))
    return mi


def compute_cmi(x, y, z, n_neighbors=3, noise_type=None):
    """Compute conditional mutual information I(x;y|z)

    :param x: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param y: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param z: real ndarray, shape (n_samples,) or (n_samples, n_features)
    :param n_neighbors: Number of nearest neighbors
    :param noise_type: add noise of given type (uniform, normal)
    :returns: non-negative estimate of conditional mutual information

    """
    n_samples = len(x)
    x, y, z = [preprocess_data(t) for t in [x, y, z]]

    if noise_type:
        rng = default_rng()
        x, y, z = [add_noise(t, rng, noise_type) for t in [x, y, z]]

    xyz = np.hstack((x, y, z))
    k = np.full(n_samples, n_neighbors)
    radius = get_radius_kneighbors(xyz, n_neighbors)

    if noise_type is None:
        # Where radius is 0, determine multiplicity
        mask = (radius == 0)
        if mask.sum() > 0:
            vals, ix, counts = np.unique(xyz[mask], axis=0,
                                         return_inverse=True,
                                         return_counts=True)
            k[mask] = counts[ix] - 1

    nxz = num_points_within_radius(np.hstack((x, z)), radius)
    nyz = num_points_within_radius(np.hstack((y, z)), radius)
    nz = num_points_within_radius(z, radius)

    cmi = max(0, np.mean(digamma(k)) - np.mean(digamma(nxz + 1))
              - np.mean(digamma(nyz + 1)) + np.mean(digamma(nz + 1)))
    return cmi


def compute_batch_mi(x, y, n_neighbors=3, noise_type=None):
    N = len(x)
    batch_size = 500
    n_batches = N//batch_size
    mi = np.zeros(n_batches)

    for i in range(n_batches):
        i0 = i * batch_size
        i1 = i0 + batch_size
        mi[i] = compute_mi(x[i0:i1], y[i0:i1], n_neighbors, noise_type)

    return mi.mean()
