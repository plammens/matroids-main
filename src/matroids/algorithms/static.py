"""The greedy algorithm for finding the maximal independent set of a matroid"""
import itertools
import typing

from matroids.matroid import Matroid, T


def maximal_independent_set(matroid: Matroid[T]) -> typing.Set[T]:
    """
    Use the greedy algorithm to find the maximal independent set of a matroid.

    :param matroid: Weighted matroid of which to find the maximal independent set.
    :return: The maximal independent set of the given matroid.
    """
    # sort elements in descending order of weight
    elements = sorted(matroid.ground_set, key=matroid.get_weight, reverse=True)

    current_set: typing.Set[T] = set()
    independence_checker = matroid.is_independent_incremental_stateful(current_set)
    # try to add elements with non-negative weight in descending order of weight
    for element in itertools.takewhile(lambda x: matroid.get_weight(x) >= 0, elements):
        if independence_checker.would_be_independent_after_adding(element):
            independence_checker.add_element(element)

    # the set is modified in-place by the ``independence_checker`` generator
    return current_set
