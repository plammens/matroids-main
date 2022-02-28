"""
Tests for the various matroid algorithms in :module:`matroids.algorithms`.
"""

import itertools as itt
import random
import typing as tp

import more_itertools as mitt
import networkx as nx
import numpy as np
import pytest

from matroids.algorithms.dynamic import (
    DynamicMaximalIndependentSetAlgorithm,
    PartialDynamicMaximalIndependentSetAlgorithm,
    RestartGreedy,
    dynamic_removal_maximal_independent_set,
    dynamic_removal_maximal_independent_set_uniform_weights,
)
from matroids.algorithms.static import (
    maximal_independent_set,
    maximal_independent_set_uniform_weights,
)
from matroids.matroid import GraphicalMatroid, RealLinearMatroid, set_weights
from matroids.matroid.uniform import IntUniformMatroid


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
    matroid = IntUniformMatroid(size=3, rank=3, weights={0: 1.0, 1: 1.0, 2: -2.0})
    result = maximal_independent_set(matroid)
    # the maximal independent set shouldn't contain the element with negative weight
    assert result == {0, 1}


def test_dynamicRemovalMaximalIndependentSet_basicSequence_correct():
    graph = nx.complete_graph(4)
    weights = {(0, 1): 2.0, (2, 3): 4.5, (1, 2): -1.0}
    set_weights(graph, weights)

    matroid = GraphicalMatroid(graph)
    remover = dynamic_removal_maximal_independent_set(matroid)

    # initial MIS
    maximal = remover.send(None)
    assert len(maximal) == 3
    assert {(2, 3), (0, 1)}.issubset(maximal) and (1, 2) not in maximal

    # remove highest weight element
    maximal = remover.send((2, 3))
    assert len(maximal) == 3
    assert (0, 1) in maximal and (1, 2) not in maximal

    # remove another element
    maximal = remover.send((0, 1))
    assert maximal == {(0, 2), (0, 3), (1, 3)}

    # remove another element, now only two edges with non-negative weight remain
    maximal = remover.send((1, 3))
    assert maximal == {(0, 3), (0, 2)}

    # remove all remaining elements, check that generator closes
    for edge in {(0, 3), (0, 2), (1, 2)}:
        maximal = remover.send(edge)
    assert maximal == set()
    with pytest.raises(StopIteration):
        remover.send(None)


@pytest.mark.parametrize(
    "algorithm",
    [
        dynamic_removal_maximal_independent_set,
        dynamic_removal_maximal_independent_set_uniform_weights,
    ],
)
def test_dynamicRemovalMaximalIndependentSet_uniformWeightsBasicSequence_correct(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm,
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


@pytest.mark.parametrize(
    "algorithm",
    [
        dynamic_removal_maximal_independent_set,
        dynamic_removal_maximal_independent_set_uniform_weights,
    ],
)
def test_dynamicRemovalMaximalIndependentSet_uniformWeightsRandomGraph_correct(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm,
):
    graph = nx.gnp_random_graph(n=50, p=0.2)
    matroid = GraphicalMatroid(graph)
    remover = algorithm(matroid)

    # select sequence of elements to remove
    sequence = list(matroid.ground_set)
    random.shuffle(sequence)
    sequence.insert(0, None)

    # use the static algorithm as a correct reference for comparison
    for to_remove in sequence:
        result_set = remover.send(to_remove)
        reference_set = maximal_independent_set(matroid)
        assert to_remove not in result_set
        assert matroid.is_independent(result_set)
        # weights are uniform so only need to check size of set to check maximality
        assert len(result_set) == len(reference_set)
