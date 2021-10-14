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
    def is_independent(self, subset: typing.AbstractSet[T]) -> bool:
        """
        Membership function for the collection of independent sets, I.

        :param subset: A subset of elements of the ground set. Behaviour is undefined
            if it contains some elements that are not in the ground set.
        :return: Whether ``subset`` is an independent set in this matroid.
        """
        pass

    def is_independent_incremental(
        self, independent_subset: typing.AbstractSet[T], new_element: T
    ) -> bool:
        """
        Special implementation of the independence check for subsets of the form S + e.

        Given a set which is already known to be independent, and an element to add to
        the set, determine whether the set formed by adding the latter to the former
        is independent.

        The purpose of this method is to allow certain subclasses to provide a
        specialized implementation for this special case that is more efficient than
        the direct independence test. The default implementation falls back on
        :meth:`is_independent`.

        :param independent_subset: A subset of elements which, as a precondition,
            is known to be independent.
        :param new_element: An element to add to ``independent_subset``. As a
            precondition, this element must not already be in ``independent_subset``.

        Important note: the preconditions are assumed and are not checked.

        :return: Whether the set formed by adding ``new_element`` to
            ``independent_subset`` is still independent.
        """
        return self.is_independent(independent_subset | {new_element})

    def is_independent_incremental_stateful(
        self, independent_subset: typing.MutableSet[T]
    ) -> typing.Generator[bool, T, None]:
        """
        Special stateful implementation of an incremental independence check.

        This method provides an implementation of the independence check in cases in
        which we want to test a sequence of subsets (S0, S1, ..., Sn, ...) such that
        each successive set is obtained by adding a new element to the previous.

        The interface is as follows. This method takes a (mutable) set S, which is
        assumed to be independent, and returns a generator object. The initial value
        of S acts as the "seed" for the sequence (S0 above), and is usually the empty
        set. When the ``.send(x)`` method of the generator is called, the generator
        tests whether S + x is independent; if it is, it adds x to S. Then,
        the generator yields whether the updated S is an independent set (as a
        boolean).

        The given set is modified in-place, so the caller will be able to see the
        modifications on the set S.

        The generator itself is infinite, so it will be disposed of only when the
        caller calls the ``.close()`` method or when it is garbage-collected.

        The purpose of this method is to allow certain subclasses to provide a
        specialized implementation for this special case that is more efficient than
        the direct independence test or successive uses of
        :meth:`is_independent_incremental`. The default implementation falls back on
        :meth:`is_independent_incremental`.

        In particular, this special case is used in the greedy algorithm for finding
        a maximal independent set.

        :param independent_subset: The subset of elements S which is modified as
            described above. As a precondition, it must be an independent set. Its
            initial value serves as the starting point for the sequence of sets.

        :return: A generator object with the above specification.
        """
        is_independent = True
        while True:
            new_element = yield is_independent
            if is_independent := self.is_independent_incremental(
                independent_subset, new_element
            ):
                independent_subset.add(new_element)

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
