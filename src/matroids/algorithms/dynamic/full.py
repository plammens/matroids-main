"""Algorithms for dynamic MIS that handle both addition and removal of elements."""

import abc
import typing

from matroids.matroid import MutableMatroid, T


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
        self.matroid = matroid

    @property
    @abc.abstractmethod
    def current(self) -> typing.Set[T]:
        """Get the current maximal independent set (without recomputing it)."""
        pass

    @abc.abstractmethod
    def add(self, element) -> typing.Set[T]:
        """
        Add an element to the matroid and return the new maximal independent set.

        :param element: Element to add.
        :return: The new maximal independent set after adding the given element.
        """
        pass

    @abc.abstractmethod
    def remove(self, element) -> typing.Set[T]:
        """
        Remove an element from the matroid and return the new maximal independent set.

        :param element: Element to remove.
        :return: The new maximal independent set after removing the given element.
        """
        pass
