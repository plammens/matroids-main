import typing

import networkx
import numpy as np

from matroids.matroid import ExplicitMatroid, Matroid, RealLinearMatroid

from matroids.algorithms.greedy import maximal_independent_set
from matroids.matroid.graphical import GraphicalMatroid
from matroids.utils import generate_subsets


def get_independent_sets(matroid: Matroid) -> typing.FrozenSet:
    """
    Compute the explicit family of independent sets, I, of the given matroid.

    Note: this function has complexity at least 2^n, where n is the size of the
    ground set. Only advised for very small instances.
    """
    return frozenset(
        subset
        for subset in generate_subsets(matroid.ground_set)
        if matroid.is_independent(subset)
    )


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


def test_graphical_matroid():
    graph = networkx.complete_graph(3)
    matroid = GraphicalMatroid(graph)
    independent_sets = get_independent_sets(matroid)
    # all and only subsets of edges of size < 3 should form an a acyclic graph
    assert independent_sets == frozenset(generate_subsets(matroid.ground_set, range(3)))


def test_negative_weights():
    # uniform matroid with three elements, one with negative weight
    matroid = ExplicitMatroid.uniform(range(3), k=3, weights={0: 1.0, 1: 1.0, 2: -2.0})
    result = maximal_independent_set(matroid)
    # the maximal independent set shouldn't contain the element with negative weight
    assert result == {0, 1}
