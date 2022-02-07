# Author: Nikolay Mayorov <n59_ru@hotmail.com>
# License: 3-clause BSD
#
# Modified by Jannis Teunissen <jannis.teunissen@cwi.nl>
# * Added method for conditional mutual information

import numpy as np
from scipy.special import digamma

from sklearn.neighbors import NearestNeighbors, KDTree


def _compute_mi_cc(x, y, n_neighbors):
    """Compute mutual information between two continuous variables.

    Parameters
    ----------
    x, y : ndarray, shape (n_samples,)
        Samples of two continuous random variables, must have an identical
        shape.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information. If it turned out to be negative it is
        replace by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    n_samples = x.size

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric="chebyshev")
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )

    return max(0, mi)


def mutual_information_v2(x, y, n_neighbors):
    """Compute mutual information between two continuous variables.

    Parameters
    ----------
    x, y : ndarray, shape (n_samples, n_features)
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
    n_samples = len(x)

    if len(x) != len(y):
        raise ValueError('x and y should have the same length')

    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric="chebyshev")
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )

    return max(0, mi)


def _compute_cmi_ccc(x, y, z, n_neighbors):
    """Compute mutual information between three continuous variables.

    Parameters
    ----------
    x, y, z : ndarray, shape (n_samples,)
        Samples of three continuous random variables, must have an identical
        shape.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated conditional mutual information.

    """
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))
    xz = np.hstack((x, z))
    yz = np.hstack((y, z))
    xyz = np.hstack((x, y, z))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)

    nn.fit(xyz)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(xz, metric="chebyshev")
    nxz = kd.query_radius(xz, radius, count_only=True, return_distance=False)
    nxz = np.array(nxz) - 1.0

    kd = KDTree(yz, metric="chebyshev")
    nyz = kd.query_radius(yz, radius, count_only=True, return_distance=False)
    nyz = np.array(nyz) - 1.0

    kd = KDTree(z, metric="chebyshev")
    nz = kd.query_radius(z, radius, count_only=True, return_distance=False)
    nz = np.array(nz) - 1.0

    mi = (
        digamma(n_neighbors)
        - np.mean(digamma(nxz + 1))
        - np.mean(digamma(nyz + 1))
        + np.mean(digamma(nz + 1))
    )

    return mi
