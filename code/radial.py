import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform


def arc_length_distance(x: NDArray, y: NDArray):
    assert x.ndim == y.ndim == 1, "both x and y should be vectors"
    assert x.shape == y.shape == (3,), "x and y should be 3-dimensional"

    chord_length = np.linalg.norm(x - y)
    arc_length = 2 * np.abs(np.arcsin(chord_length / 2))
    return arc_length


def fast_pairwise_arc_length_distance(X: NDArray):
    assert X.ndim == 2

    diffs = X[:, None, :] - X[None, :, :]
    chord_lengths = squareform(np.linalg.norm(diffs, axis=-1), 'tovector', checks=True)
    arc_lengths = 2 * np.abs(np.arcsin(chord_lengths / 2))

    return arc_lengths


def pairwise_arc_length_distance(X: NDArray):
    assert X.ndim == 2

    dist = pdist(X, arc_length_distance)
    return dist
