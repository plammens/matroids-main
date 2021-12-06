import typing

import networkx
import networkx as nx
import numpy as np
import pytest

from matroids.matroid import ExplicitMatroid, Matroid, RealLinearMatroid

from matroids.algorithms.static import (
    maximal_independent_set,
    maximal_independent_set_uniform_weights,
)
from matroids.algorithms.dynamic import dynamic_maximal_independent_set_remove
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


def test_basic_maximal_independent_set_uniform_weights():
    matrix = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    matroid = RealLinearMatroid(matrix)  # uniform weights
    result = maximal_independent_set_uniform_weights(matroid)
    # maximal set of l.i. columns is 2nd and 3rd
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


def test_dynamic_maximal_independent_set_remove():
    graph = nx.complete_graph(4)
    weights = {(0, 1): 2.0, (2, 3): 4.5, (1, 2): -1.0}
    for (u, v), w in weights.items():
        graph[u][v]["weight"] = w

    matroid = GraphicalMatroid(graph)
    generator = dynamic_maximal_independent_set_remove(matroid)

    # initial MIS
    maximal = generator.send(None)
    assert len(maximal) == 3
    assert {(2, 3), (0, 1)}.issubset(maximal) and (1, 2) not in maximal

    # remove highest weight element
    maximal = generator.send((2, 3))
    assert len(maximal) == 3
    assert (0, 1) in maximal and (1, 2) not in maximal

    # remove another element
    maximal = generator.send((0, 1))
    assert len(maximal) == 3
    assert (1, 2) not in maximal

    # remove another element, now only two edges with non-negative weight remain
    maximal = generator.send((1, 3))
    assert maximal == {(0, 3), (0, 2)}

    # remove all remaining elements, check that generator closes
    for edge in {(0, 3), (0, 2), (1, 2)}:
        maximal = generator.send(edge)
    assert maximal == set()
    with pytest.raises(StopIteration):
        generator.send(None)


def test_naive_dynamic_remove_uniform_weights():
    # test that removing an element from a matroid with uniform weights
    # selects the appropriate element to restart from in naive dynamic algorithm
    # (searching by weight isn't enough, as weight isn't unique)

    graph = nx.complete_graph(5)
    matroid = GraphicalMatroid(graph)
    remover = dynamic_maximal_independent_set_remove(matroid)
    maximal = remover.send(None)

    to_remove = list(maximal)[2]
    maximal = remover.send(to_remove)
    assert maximal == maximal_independent_set(matroid)
