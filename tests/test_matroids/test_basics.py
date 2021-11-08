import typing

import networkx
import numpy as np

from matroids.matroid import ExplicitMatroid, Matroid, RealLinearMatroid

from matroids.algorithms.greedy import maximal_independent_set
from matroids.matroid.graphical import GraphicalMatroid
from matroids.utils import generate_subsets


def get_independent_sets(matroid: Matroid) -> typing.FrozenSet[typing.FrozenSet]:
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
    graph = networkx.cycle_graph(4)
    matroid = GraphicalMatroid(graph)
    independent_sets = get_independent_sets(matroid)
    # only subsets of edges of size < 3, plus some of size 4, form an a acyclic graph
    assert independent_sets == (
        frozenset(generate_subsets(matroid.ground_set, range(3)))
        | frozenset(
            {
                # all edges minus one
                frozenset({(0, 1), (1, 2), (2, 3)}),
                frozenset({(0, 1), (1, 2), (0, 3)}),
                frozenset({(0, 1), (2, 3), (0, 3)}),
                frozenset({(1, 2), (2, 3), (0, 3)}),
            }
        )
    )


def test_negative_weights():
    # uniform matroid with three elements, one with negative weight
    matroid = ExplicitMatroid.uniform(range(3), k=3, weights={0: 1.0, 1: 1.0, 2: -2.0})
    result = maximal_independent_set(matroid)
    # the maximal independent set shouldn't contain the element with negative weight
    assert result == {0, 1}


def test_add_element():
    matroid = ExplicitMatroid.uniform(range(3), k=2)
    matroid.add_element(5)
    assert matroid.ground_set == frozenset({0, 1, 2, 5})
    assert matroid.weights[5] == 1.0


def test_remove_element():
    matroid = ExplicitMatroid.uniform(range(3), k=2)
    matroid.remove_element(1)
    assert matroid.ground_set == frozenset({0, 2})
    assert matroid.independent_sets == set(map(frozenset, [{}, {0}, {2}, {0, 2}]))
    assert 1 not in matroid.weights
