import typing as tp

import numpy as np

from matroids.matroid import MutableMatroid, T
from .static import maximal_independent_set_uniform_weights


# type alias for dynamic maximal independent set algorithms:
# takes a mutable matroid as an argument
# and returns a generator that accepts elements to add/remove and yields the maximal
# independent set after removing the given element from the matroid
DynamicMaximalIndependentSetAlgorithm = tp.Callable[
    [MutableMatroid[T]], tp.Generator[tp.Set, T, None]
]


def dynamic_removal_maximal_independent_set(
    matroid: MutableMatroid[T],
) -> tp.Generator[tp.Set, T, None]:
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
        for j in range(i, len(elements)):
            element = elements[j]
            if element in removed:
                continue
            indicators[j] = independence_checker.add_if_independent(element)
        del independence_checker  # dispose of checker

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


def dynamic_addition_maximal_independent_set_uniform_weights(
    matroid: MutableMatroid[T],
) -> tp.Generator[tp.Set, T, None]:
    """
    Compute the M.I.S. after each addition of an element, assuming uniform weights.

    Assumes that the weight of all elements is equal, i.e. all elements have the same
    positive weight.

    :param matroid: Uniformly weighted matroid of which to compute the maximal
        independent set.
    :return: A generator that accepts elements to add and yields the maximal
        independent set after removing the given element from the matroid.
    """
    # compute initial M.I.S.
    current_set = maximal_independent_set_uniform_weights(matroid)

    # since all weights are the same, adding an element is just a matter of independence
    independence_checker = matroid.is_independent_incremental_stateful(current_set)
    while True:
        new_element = yield current_set
        independence_checker.add_if_independent(new_element)


def dynamic_removal_maximal_independent_set_uniform_weights(
    matroid: MutableMatroid[T],
) -> tp.Generator[tp.Set, T, None]:
    """
    Compute the M.I.S. after each removal of an element, assuming uniform weights.

    Assumes that the weight of all elements is equal, i.e. all elements have the same
    positive weight.

    :param matroid: Uniformly weighted matroid of which to compute the maximal
        independent set.
    :return: A generator that accepts elements to remove and yields the maximal
        independent set after removing the given element from the matroid.
    """

    # ----------- do static algorithm to get initial MIS -----------
    current_set = set()
    checker = matroid.is_independent_incremental_stateful(current_set)
    ground_set_nonzero = frozenset(
        filter(checker.would_be_independent_after_adding, matroid.ground_set)
    )
    elements = set(ground_set_nonzero)

    # list of pivot elements (elements added to MIS)
    pivots: tp.List[T] = []
    # witness set i: elements removed at pivot step i
    witness_sets: tp.List[tp.Set[T]] = []

    while elements:
        # select arbitrary pivot element to add to the independent set
        pivot = elements.pop()
        checker.add_element(pivot)
        pivots.append(pivot)

        # compute witness set
        to_remove = {
            element
            for element in elements
            if not checker.would_be_independent_after_adding(element)
        }
        elements.difference_update(to_remove)
        witness_sets.append(to_remove)

    del checker

    # --------- dynamic part ---------------
    while not matroid.is_empty:
        element_to_remove = yield current_set

        # check if the element is a pivot
        if element_to_remove in pivots:
            # is a pivot; must restart algorithm from that step
            step = pivots.index(element_to_remove)

            # reset pivots and witness steps from the step
            del pivots[step:]
            del witness_sets[step:]

            # reconstruct current independent set
            current_set = set(pivots)
            checker = matroid.is_independent_incremental_stateful(current_set)

            # reconstruct available elements
            elements = set(ground_set_nonzero)  # make copy O(n)
            # remove discarded elements up to the given step
            elements.difference_update(*witness_sets)

            while elements:
                # select arbitrary pivot element to add to the independent set
                pivot = elements.pop()
                checker.add_element(pivot)
                pivots.append(pivot)

                # compute witness set
                to_remove = {
                    element
                    for element in elements
                    if not checker.would_be_independent_after_adding(element)
                }
                elements.difference_update(to_remove)
                witness_sets.append(to_remove)

            del checker

        else:
            # not a pivot; solution doesn't change, only need to update witness sets
            for witness in witness_sets:
                witness.discard(element_to_remove)
