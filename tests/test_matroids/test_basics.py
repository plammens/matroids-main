import numpy as np

from matroids.matroid import RealLinearMatroid

from matroids.algorithms.greedy import maximal_independent_set


def test_basic_maximal_independent_set():
    matrix = np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    weights = np.array([2, 3, 1])
    matroid = RealLinearMatroid(matrix, weights)
    result = maximal_independent_set(matroid)
    # should have selected 2nd and 3rd columns:
    assert result == {1, 2}
