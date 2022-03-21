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

from matroids.algorithms.dynamic.full import (
    DynamicMaximalIndependentSetAlgorithm,
    DynamicMaximalIndependentSetComputer,
    NaiveDynamic,
    RestartGreedy,
)
from matroids.algorithms.dynamic.partial import (
    PartialDynamicMaximalIndependentSetAlgorithm,
    dynamic_removal_maximal_independent_set_uniform_weights,
)
from matroids.algorithms.static import (
    maximal_independent_set,
    maximal_independent_set_uniform_weights,
)
from matroids.matroid import (
    GraphicalMatroid,
    IntUniformMatroid,
    MutableIntUniformMatroid,
    MutableMatroid,
    RealLinearMatroid,
    set_weights,
)
from matroids.utils import RandomAccessMutableSet


DYNAMIC_REMOVAL_ALGORITHMS = [
]

DYNAMIC_REMOVAL_UNIFORM_WEIGHTS_ALGORITHMS = [
    dynamic_removal_maximal_independent_set_uniform_weights,
]

FULL_DYNAMIC_ALGORITHMS = [
    RestartGreedy,
    NaiveDynamic,
]


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


@pytest.mark.parametrize("algorithm", DYNAMIC_REMOVAL_ALGORITHMS)
def test_dynamicRemovalMaximalIndependentSet_basicSequence_correct(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm,
):
    graph = nx.complete_graph(4)
    weights = {(0, 1): 2.0, (2, 3): 4.5, (1, 2): -1.0}
    set_weights(graph, weights)

    matroid = GraphicalMatroid(graph)
    remover = algorithm(matroid)

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


@pytest.mark.parametrize("algorithm", DYNAMIC_REMOVAL_ALGORITHMS)
def test_dynamicRemovalMaximalIndependentSet_randomGraph_correct(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm,
):
    _test_dynamicRemovalMaximalIndependentSet_randomGraph(
        algorithm, uniform_weights=False
    )


@pytest.mark.parametrize("algorithm", DYNAMIC_REMOVAL_UNIFORM_WEIGHTS_ALGORITHMS)
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


@pytest.mark.parametrize("algorithm", DYNAMIC_REMOVAL_UNIFORM_WEIGHTS_ALGORITHMS)
def test_dynamicRemovalMaximalIndependentSet_uniformWeightsRandomGraph_correct(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm,
):
    _test_dynamicRemovalMaximalIndependentSet_randomGraph(
        algorithm, uniform_weights=True
    )


@pytest.mark.parametrize("algorithm", FULL_DYNAMIC_ALGORITHMS)
def test_fullDynamicMaximalIndependentSet_basicSequence1_correct(
    algorithm: DynamicMaximalIndependentSetAlgorithm,
):
    graph = nx.complete_graph(4)
    weights = {(0, 1): 2.0, (2, 3): 4.5, (1, 2): -1.0}
    set_weights(graph, weights)

    matroid = GraphicalMatroid(graph)
    algorithm_instance = algorithm(matroid)

    # initial MIS
    current_maximal = algorithm_instance.current
    assert len(current_maximal) == 3
    assert {(2, 3), (0, 1)}.issubset(current_maximal) and (1, 2) not in current_maximal

    # remove highest weight element
    current_maximal = algorithm_instance.remove_element((2, 3))
    assert len(current_maximal) == 3
    assert (0, 1) in current_maximal and (1, 2) not in current_maximal

    # add an element of negative weight; shouldn't change
    assert algorithm_instance.add_element((2, 3), weight=-1.0) == current_maximal

    # remove another element
    assert algorithm_instance.remove_element((0, 1)) == {(0, 2), (0, 3), (1, 3)}

    # remove another element, now only two edges with non-negative weight remain
    assert algorithm_instance.remove_element((1, 3)) == {(0, 3), (0, 2)}

    # re-add an edge
    assert algorithm_instance.add_element((0, 1)) == {(0, 1), (0, 2), (0, 3)}

    # add another edge with greater weight
    previous_maximal = algorithm_instance.current
    current_maximal = algorithm_instance.add_element((1, 3), weight=2.0)
    assert current_maximal - previous_maximal == {(1, 3)}
    assert len(previous_maximal - current_maximal) == 1


@pytest.mark.parametrize("algorithm", FULL_DYNAMIC_ALGORITHMS)
def test_fullDynamicMaximalIndependentSet_basicSequence2_correct(
    algorithm: tp.Type[DynamicMaximalIndependentSetComputer],
):
    # start with an empty matroid
    matroid = MutableIntUniformMatroid(size=0, rank=3)
    algorithm_instance = algorithm(matroid)

    assert algorithm_instance.current == set()

    assert algorithm_instance.add_element(1) == {1}

    # re-adding the same element shouldn't have an effect
    assert algorithm_instance.add_element(1) == {1}

    assert algorithm_instance.add_element(2) == {1, 2}

    # negative weights should be ignored
    assert algorithm_instance.add_element(3, weight=-1.0) == {1, 2}

    assert algorithm_instance.add_element(4) == {1, 2, 4}

    # if we add another element, the MIS can't increase, since the rank is 3
    previous_maximal = algorithm_instance.current
    current_maximal = algorithm_instance.add_element(5)
    if current_maximal != previous_maximal:
        # check that 5 was exchanged for another element
        assert current_maximal - previous_maximal == {5}
        assert len(previous_maximal - current_maximal) == 1

    # add another element of greater weight
    previous_maximal = algorithm_instance.current
    current_maximal = algorithm_instance.add_element(6, weight=2.0)
    # now an update is forced; check that 6 was exchanged for another element
    assert len(current_maximal) == 3
    assert 6 in current_maximal

    # update the weight of an existing element (3 had negative weight before)
    current_maximal = algorithm_instance.add_element(3, weight=100.0)
    assert len(current_maximal) == 3
    assert 3 in current_maximal

    # remove and update some elements
    algorithm_instance.remove_element(1)
    algorithm_instance.remove_element(2)
    algorithm_instance.remove_element(3)
    algorithm_instance.add_element(4, weight=-1.0)
    assert algorithm_instance.current == {5, 6}


@pytest.mark.parametrize("algorithm", FULL_DYNAMIC_ALGORITHMS)
def test_fullDynamicMaximalIndependentSet_basicSequence3_correct(
    algorithm: DynamicMaximalIndependentSetAlgorithm,
):
    graph = nx.Graph()
    matroid = GraphicalMatroid(graph)
    algorithm_instance = algorithm(matroid)

    algorithm_instance.add_element((0, 1))
    algorithm_instance.add_element((0, 2))
    algorithm_instance.add_element((1, 2))

    # check that when rebuilding the MIS, previous greedy work is remembered
    # (only elements that were previously selected are selected)
    previous_maximal = algorithm_instance.current
    current_maximal = algorithm_instance.add_element((3, 4), weight=0.5)
    assert current_maximal == previous_maximal | {(3, 4)}


@pytest.mark.parametrize("algorithm", FULL_DYNAMIC_ALGORITHMS)
def test_fullDynamicMaximalIndependentSet_randomGraph(
    algorithm: DynamicMaximalIndependentSetAlgorithm,
):
    _test_fullDynamicMaximalIndependentSet_randomGraph(algorithm)


def _test_dynamicRemovalMaximalIndependentSet_randomGraph(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm, uniform_weights: bool
):
    matroid = _random_graphical_matroid(uniform_weights)
    remover = algorithm(matroid)

    # select sequence of elements to remove
    sequence = list(matroid.ground_set)
    random.shuffle(sequence)
    sequence.insert(0, None)

    # use the static algorithm as a correct reference for comparison
    for to_remove in sequence:
        _check_removal(remover.send, matroid, to_remove)


def _test_fullDynamicMaximalIndependentSet_randomGraph(
    algorithm: DynamicMaximalIndependentSetAlgorithm, uniform_weights: bool = False
):
    matroid = _random_graphical_matroid(uniform_weights)
    algorithm_instance = algorithm(matroid)

    # store elements as a sequence for use with random.choice
    possible_elements_to_add = list(matroid.ground_set)
    possible_elements_to_remove = RandomAccessMutableSet(matroid.ground_set)

    # perform a certain number of additions / deletions at random
    for _ in range(100):
        action = random.choice(["add", "remove"])
        if action == "add":
            to_add = random.choice(possible_elements_to_add)
            weight = random.uniform(-1.0, 1.0)
            _check_addition(algorithm_instance.add_element, matroid, to_add, weight)
            possible_elements_to_remove.add(to_add)
        elif action == "remove":
            to_remove = random.choice(possible_elements_to_remove)
            _check_removal(algorithm_instance.remove_element, matroid, to_remove)
            possible_elements_to_remove.remove(to_remove)
        else:
            assert False


def _random_graphical_matroid(uniform_weights: bool):
    graph = nx.gnp_random_graph(n=50, p=0.2)
    if not uniform_weights:
        weights = {edge: random.uniform(-1.0, 1.0) for edge in graph.edges}
        set_weights(graph, weights)
    return GraphicalMatroid(graph)


def _check_addition(
    adder_function: tp.Callable[[tp.Any, float], tp.AbstractSet],
    matroid: MutableMatroid,
    element_to_add,
    weight: float,
):
    """
    Compare an addition algorithm with the static algorithm for correctness.

    :param adder_function: Function taking the element to add and its weight,
        which adds the element to the matroid and returns the updated solution.
    :param matroid: Matroid under test.
    :param element_to_add: Element to add.
    :param weight: Weight of the element to add.
    """
    previous_ground_set = set(matroid.ground_set)
    result_set = adder_function(element_to_add, weight)
    assert matroid.ground_set == previous_ground_set | {element_to_add}

    # compute correct solution
    reference_set = maximal_independent_set(matroid)

    assert matroid.is_independent(result_set)
    assert len(result_set) == len(reference_set)
    assert matroid.total_weight(result_set) == matroid.total_weight(reference_set)


def _check_removal(
    remover_function: tp.Callable[[tp.Any], tp.AbstractSet],
    matroid: MutableMatroid,
    element_to_remove,
):
    """
    Compare a removal algorithm with the static algorithm for correctness.

    :param remover_function: Function taking the element to add and its weight,
        which adds the element to the matroid and returns the updated solution.
    :param matroid: Matroid under test.
    :param element_to_remove: Element to remove.
    """
    previous_ground_set = set(matroid.ground_set)
    result_set = remover_function(element_to_remove)
    assert matroid.ground_set == previous_ground_set - {element_to_remove}

    # compute correct solution
    reference_set = maximal_independent_set(matroid)

    assert element_to_remove not in result_set
    assert matroid.is_independent(result_set)
    assert len(result_set) == len(reference_set)
    assert matroid.total_weight(result_set) == matroid.total_weight(reference_set)
