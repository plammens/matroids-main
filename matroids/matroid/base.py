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
    """

    @property
    @abc.abstractmethod
    def ground_set(self) -> typing.Set[T]:
        """Returns the ground set E corresponding to this matroid."""
        pass

    @abc.abstractmethod
    def is_independent(self, subset: typing.Set[T]) -> bool:
        """
        Membership function for the collection of independent sets, I.

        :param subset: A subset of elements of the ground set.
        :return: Whether ``subset`` is an independent set in this matroid.
        """
        pass


class WeightedMatroid(Matroid[T], metaclass=abc.ABCMeta):
    """
    Abstract base class for weighted matroids.

    Extends regular matroids by associating a numerical weight to each element of the
    ground set.
    """

    @abc.abstractmethod
    def get_weight(self, element: T) -> float:
        """Returns the weight of the given element."""
        pass
