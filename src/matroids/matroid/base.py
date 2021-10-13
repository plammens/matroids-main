import abc

import typing


T = typing.TypeVar("T")


class Matroid(typing.Generic[T], metaclass=abc.ABCMeta):
    """
    Abstract base class for matroids.

    A matroid is a tuple (E, I) where E is a set, and I is a collection of subsets of E
    (called "independent sets") such that:
       (1) for all X in I, if Y is a subset of X, then X is also in I
       (2) for all X, Y in I, if |Y| > |X| then there is a y in Y but not in X such that
           X union {y} is also in I.

    The type variable T corresponds to the type of the elements of the ground set E.

    Some matroids may assign custom numeric weights to each element of the ground set
    via the :meth:`get_weight` method.
    """

    @property
    @abc.abstractmethod
    def ground_set(self) -> typing.FrozenSet[T]:
        """Returns the ground set E corresponding to this matroid."""
        pass

    @abc.abstractmethod
    def is_independent(self, subset: typing.Collection[T]) -> bool:
        """
        Membership function for the collection of independent sets, I.

        :param subset: A subset of elements of the ground set.
        :return: Whether ``subset`` is an independent set in this matroid.
        """
        pass

    @abc.abstractmethod
    def get_weight(self, element: T) -> float:
        """
        Returns the weight of the given element.

        Subclasses should override this if they need to implement the ability to assign
        custom weights to each element. By default, all elements are assigned
        unit weight.

        :param element: Element of the ground set whose weight to get.
        :return: The numeric weight associated to the given element.
        """
        return 1.0
