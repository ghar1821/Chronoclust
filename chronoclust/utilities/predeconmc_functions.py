import numpy as np
from numba import jit, prange

@jit(nopython=True)
def calculate_euclidean_dist(a, b):
    """
    Calculate Euclidean distance between two points.

    Args:
        a (numpy.array): point 1.
        b (numpy.array): point 2.

    Returns:
        Float: Euclidean Distance between 2 points.
    """
    x = np.subtract(a,b)
    return np.linalg.norm(x)

@jit(nopython=True)
def calculate_variance_along_dimension(point, neighbours):
    """
    Calculate the variance of a point based on its neighbourhood.
    The variance is only calculated for a dimension.
    So make sure the point and neighbours are values for a dimension only!
    See Definition 1 in Bohm paper.

    Args:
        point: a point's value for a dimension.
        neighbours: the point's neighbours. Only give values for a dimension!

    Returns:
        Float: Variance of neighbourhood of a point along a dimension
    """

    subtractions = np.subtract(point, neighbours)
    sqr = np.square(subtractions)
    summation = np.sum(sqr)

    num_neighbour = len(neighbours)
    variance = np.divide(summation, num_neighbour)

    return variance

@jit(nopython=True)
def calculate_weighted_dist_squared(pref_vector, p, q):
    """
    Calculate the weighted distance between 2 points (dist_p in paper[2]). See definition 3 in paper[2]

    Args:
        pref_vector (numpy array): Point 1 subspace_preference_vector
        p (numpy array): Point 1.
        q (numpy array): Point 2.

    Returns:
        Float: Weighted distance between 2 points
    """
    subtraction = np.subtract(p, q)
    sqr = np.square(subtraction)
    mult = np.multiply(pref_vector, sqr)
    weight_dist_sqr = np.sum(mult)

    return weight_dist_sqr
