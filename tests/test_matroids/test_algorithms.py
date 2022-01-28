"""
Tests for the various matroid algorithms in :module:`matroids.algorithms`.
"""

import itertools as itt

import more_itertools as mitt
import networkx as nx
import numpy as np
import pytest

from matroids.algorithms.dynamic import (
    DynamicMaximalIndependentSetAlgorithm,
    dynamic_removal_maximal_independent_set,
    dynamic_removal_maximal_independent_set_uniform_weights,
)
from matroids.algorithms.static import (
    maximal_independent_set,
    maximal_independent_set_uniform_weights,
)
from matroids.matroid import ExplicitMatroid, GraphicalMatroid, RealLinearMatroid


def test_maximalIndependentSet_linearMatroid_correct():
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


def test_maximalIndependentSet_linearMatroidUniformWeights_correct():
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


def test_maximalIndependentSet_matroidWithNegativeWeights_negativeWeightsIgnored():
    # free matroid with three elements, one with negative weight
    matroid = ExplicitMatroid.uniform(range(3), k=3, weights={0: 1.0, 1: 1.0, 2: -2.0})
    result = maximal_independent_set(matroid)
    # the maximal independent set shouldn't contain the element with negative weight
    assert result == {0, 1}


def test_dynamicRemovalMaximalIndependentSet_basicSequence_correct():
    graph = nx.complete_graph(4)
    weights = {(0, 1): 2.0, (2, 3): 4.5, (1, 2): -1.0}
    for (u, v), w in weights.items():
        graph[u][v]["weight"] = w

    matroid = GraphicalMatroid(graph)
    generator = dynamic_removal_maximal_independent_set(matroid)

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


@pytest.mark.parametrize(
    "algorithm",
    [
        dynamic_removal_maximal_independent_set,
        dynamic_removal_maximal_independent_set_uniform_weights,
    ],
)
def test_dynamicRemovalMaximalIndependentSet_uniformWeightsBasicSequence_correct(
    algorithm: DynamicMaximalIndependentSetAlgorithm,
):
    graph = nx.complete_graph(4)
    matroid = GraphicalMatroid(graph)
    remover = algorithm(matroid)

    # initial MIS
    maximal = remover.send(None)
    assert len(maximal) == 3
    assert matroid.is_independent(maximal)

    # remove an element not in maximal
    to_remove = mitt.first(itt.filterfalse(maximal.__contains__, matroid.ground_set))
    new_maximal = remover.send(to_remove)
    assert new_maximal == maximal
    maximal = new_maximal
    del new_maximal

    # remove elements in maximal
    for to_remove in list(maximal):
        maximal = remover.send(to_remove)
        assert to_remove not in maximal
        assert matroid.is_independent(maximal)
        # check size approximately
        assert len(maximal) >= min(3, len(matroid.ground_set) - 1)
