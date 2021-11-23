import typing

import numpy as np

from matroids.matroid import MutableMatroid, T


def dynamic_maximal_independent_set_remove(
    matroid: MutableMatroid[T],
) -> typing.Generator[typing.Set, T, None]:
    """
    Compute the maximal independent set after each removal of an element.

    :param matroid: Weighted matroid of which to find the maximal independent set.
    :return: A generator that accepts elements to remove and yields the maximal
        independent set after removing the given element from the matroid.
    """
    # sort elements in descending order of weight
    elements = sorted(
        filter(lambda x: matroid.get_weight(x) >= 0, matroid.ground_set),
        key=matroid.get_weight,
        reverse=True,
    )
    # array of boolean indicators for whether each element is in the MIS at the ith step
    indicators = np.zeros(shape=(len(elements),), dtype=bool)

    # array of weights (in ascending order) for calls to np.searchsorted
    ascending_weights = np.array([matroid.get_weight(x) for x in elements])[::-1]

    removed = set()
    i = 0  # sorted index of removed element
    while not matroid.is_empty:
        # recover greedy algorithm set just before adding the deleted element
        current_set = set(
            element
            for element, flag in zip(elements[:i], indicators[:i])
            if flag and element not in removed
        )

        # rerun greedy from this point onwards
        independence_checker = matroid.is_independent_incremental_stateful(current_set)
        independence_checker.send(None)  # start generator
        for j in range(i, len(elements)):
            element = elements[j]
            if element in removed:
                continue
            indicators[j] = independence_checker.send(element)
        independence_checker.close()  # dispose of generator

        # reuse the current MIS while the element to remove is not in it
        still_valid = True
        while still_valid:
            to_remove = yield current_set
            to_remove_weight = matroid.get_weight(to_remove)
            matroid.remove_element(to_remove)
            removed.add(to_remove)
            still_valid = to_remove not in current_set

        # do binary search to find index of removed element
        # Must reverse index because elements are in descending order, but
        # np.searchsorted() works on ascending order.
        i = (len(elements) - 1) - np.searchsorted(
            ascending_weights,
            to_remove_weight,  # noqa
            side="right",  # in case of equal weights, select first in descending order
        )
        # there might be elements with the same weight, do linear search from here
        while elements[i] != to_remove:  # noqa
            i += 1

    # matroid is empty; yield empty set (MIS) as the final yield, also as a sentinel
    yield set()
