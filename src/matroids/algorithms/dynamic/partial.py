"""Algorithms for dynamic MIS that handle only additions or only removals."""

import random
import typing as tp

from matroids.matroid import MutableMatroid, T
from matroids.utils import RandomAccessMutableSet
from ..static import maximal_independent_set_uniform_weights


# type alias for (partial) dynamic maximal independent set algorithms:
# takes a mutable matroid as an argument
# and returns a generator that accepts elements to add (remove) and yields the maximal
# independent set after adding (removing) the given element from the matroid
PartialDynamicMaximalIndependentSetAlgorithm = tp.Callable[
    [MutableMatroid[T]], tp.Generator[tp.Set, T, None]
]


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
    independence_checker = matroid.stateful_independence_checker(current_set)
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

    # set of available elements at ith step; starts with independent singletons
    witness_sets: tp.List[RandomAccessMutableSet[T]] = [
        RandomAccessMutableSet(
            x for x in matroid.ground_set if matroid.is_independent({x})
        )
    ]
    # elements selected for the maximal independent set
    pivots: tp.List[T] = []

    step = 0  # greedy algorithm step (index of witness set / pivot to choose)
    while not matroid.is_empty:
        # recover the set of available elements (that can be added) at the given step
        del witness_sets[(step + 1) :]
        available_elements = witness_sets[step]

        # recover greedy algorithm set just before adding the deleted element
        del pivots[step:]
        current_set = set(pivots)

        # rerun greedy algorithm from this point onwards
        independence_checker = matroid.stateful_independence_checker(current_set)
        while available_elements:
            # select arbitrary pivot element to add to the independent set
            pivot = random.choice(available_elements)
            available_elements.discard(pivot)
            independence_checker.add_element(pivot)
            pivots.append(pivot)

            # advance onto the following step
            step += 1
            # update available elements
            available_elements = RandomAccessMutableSet(
                x
                for x in available_elements
                if independence_checker.would_be_independent_after_adding(x)
            )
            # store as next witness set
            witness_sets.append(available_elements)

        # reuse the current MIS while the element to remove is not in it (not a pivot)
        still_valid = True
        while still_valid:
            element_to_remove = yield current_set
            matroid.remove_element(element_to_remove)
            # update witness sets
            for witness_set in witness_sets:
                witness_set.discard(element_to_remove)

            # check whether removed element is pivot
            still_valid = element_to_remove not in current_set

        # removing a pivot; find out algorithm step and start over
        step = pivots.index(element_to_remove)  # noqa

    # matroid is empty; yield empty set (MIS) as the final yield, also as a sentinel
    yield set()
