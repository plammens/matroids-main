"""The greedy algorithm for finding the maximal independent set of a matroid"""

import typing as tp

from matroids.matroid import Matroid, T


def maximal_independent_set(matroid: Matroid[T]) -> tp.Set[T]:
    """
    Use the greedy algorithm to find the maximal independent set of a matroid.

    :param matroid: Weighted matroid of which to find the maximal independent set.
    :return: The maximal independent set of the given matroid.
    """
    # discard elements with negative weight
    elements = filter(lambda x: matroid.get_weight(x) >= 0, matroid.ground_set)
    # sort elements by descending order of weight
    elements = sorted(elements, key=matroid.get_weight, reverse=True)

    return _greedy_core(matroid, elements)


def maximal_independent_set_uniform_weights(matroid: Matroid[T]) -> tp.Set[T]:
    """
    Compute the maximal independent set under the assumption that all weights are equal.

    Assumes as a precondition that the weight of all elements is equal, i.e. all
    elements have the same positive weight.

    :param matroid: Uniformly weighted matroid of which to compute the maximal
        independent set.
    :return: The maximal independent set of the given matroid.
    """
    # no need to sort elements as they all have the same positive weight by assumption
    return _greedy_core(matroid, matroid.ground_set)


def _greedy_core(matroid: Matroid[T], elements_iterable: tp.Iterable[T]) -> tp.Set[T]:
    """
    Core of the greedy algorithm for computing the maximal independent set.

    :param matroid: Matroid whose maximal independent set is to be found.
    :param elements_iterable: Elements of the matroid given in such an order that
        running this algorithm with this order yields the maximal independent set
        (the caller should make sure of this). Usually this would be the elements
        sorted by descending order of weight.

    :returns: The maximal independent set of the given matroid.
    """
    current_set: tp.Set[T] = set()
    independence_checker = matroid.is_independent_incremental_stateful(current_set)

    # greedy part: keep adding next element if it maintains independence
    for element in elements_iterable:
        independence_checker.add_if_independent(element)

    # the set is modified in-place by the ``independence_checker`` generator
    return current_set
