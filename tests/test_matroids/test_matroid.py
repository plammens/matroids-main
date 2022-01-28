"""
Tests for the :class:`matroids.matroid.Matroid` class hierarchy.
"""

import typing

import networkx

from matroids.matroid import ExplicitMatroid, GraphicalMatroid, Matroid
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


def test_mutableMatroid_addElement_addedCorrectly():
    matroid = ExplicitMatroid.uniform(range(3), k=2)
    matroid.add_element(5)
    assert matroid.ground_set == frozenset({0, 1, 2, 5})
    assert matroid.weights[5] == 1.0


def test_mutableMatroid_removeElement_removedCorrectly():
    matroid = ExplicitMatroid.uniform(range(3), k=2)
    matroid.remove_element(1)
    assert matroid.ground_set == frozenset({0, 2})
    assert matroid.independent_sets == set(map(frozenset, [{}, {0}, {2}, {0, 2}]))
    assert 1 not in matroid.weights


def test_graphicalMatroid_independentSets_correct():
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
