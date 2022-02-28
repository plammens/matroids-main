"""Algorithms for dynamic MIS that handle both addition and removal of elements."""

import abc
import typing as tp

from matroids.matroid import MutableMatroid, T
from ..static import maximal_independent_set


class DynamicMaximalIndependentSetAlgorithm(metaclass=abc.ABCMeta):
    """
    Interface for implementations of the dynamic maximal independent set algorithm.

    Presents three main interaction points:
     - the constructor/initializer: takes the matroid of which to compute the maximal
       independent sets, and initializes some internal state
     - the add method: takes an element to add to the matroid and returns the new
       maximal independent set
     - the remove method: takes an element to remove and returns the new maximal
       independent set

    The matroid shouldn't be mutated while it is being used in an instance of this
    class, otherwise undefined behaviour will ensue.
    """

    def __init__(self, matroid: MutableMatroid[T]):
        """
        Initialise the algorithm for a given matroid.

        :param matroid: The matroid in which elements will be added/removed and of which
            to compute the maximal independent set after each update.
        """
        self._matroid = matroid

    @property
    @abc.abstractmethod
    def current(self) -> tp.Set[T]:
        """Get the current maximal independent set (without recomputing it)."""
        pass

    @abc.abstractmethod
    def add_element(self, element: T, weight: tp.Optional[float] = None) -> tp.Set[T]:
        """
        Add an element to the matroid and return the new maximal independent set.

        :param element: Element to add.
        :param weight: Optional weight for the new element.
        :return: The new maximal independent set after adding the given element.
        """
        pass

    @abc.abstractmethod
    def remove_element(self, element) -> tp.Set[T]:
        """
        Remove an element from the matroid and return the new maximal independent set.

        :param element: Element to remove.
        :return: The new maximal independent set after removing the given element.
        """
        pass


class RestartGreedy(DynamicMaximalIndependentSetAlgorithm):
    """The baseline approach: rerun the greedy algorithm after each update."""

    def __init__(self, matroid: MutableMatroid[T]):
        super().__init__(matroid)
        self._current = maximal_independent_set(matroid)

    @property
    def current(self) -> tp.Set[T]:
        return self._current

    def add_element(self, element: T, weight: tp.Optional[float] = None) -> tp.Set[T]:
        self._matroid.add_element(element, weight=weight)
        self._current = result = maximal_independent_set(self._matroid)
        return result

    def remove_element(self, element: T) -> tp.Set[T]:
        self._matroid.remove_element(element)
        self._current = result = maximal_independent_set(self._matroid)
        return result
