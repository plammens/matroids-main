import abc

import typing as tp


T = tp.TypeVar("T")


class Matroid(tp.Generic[T], metaclass=abc.ABCMeta):
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
    def ground_set(self) -> tp.AbstractSet[T]:
        """
        Returns the set of elements (the ground set, E) of this matroid.

        The caller shouldn't attempt to mutate the returned object, otherwise
        undefined behaviour might ensue.
        """
        pass

    def __bool__(self):
        """As per the Python convention, whether the matroid is nonempty."""
        return bool(self.ground_set)

    @property
    @tp.final
    def is_empty(self):
        """Whether the matroid is empty."""
        return not bool(self)

    @abc.abstractmethod
    def is_independent(self, subset: tp.AbstractSet[T]) -> bool:
        """
        Membership function for the collection of independent sets, I.

        :param subset: A subset of elements of the ground set. Behaviour is undefined
            if it contains some elements that are not in the ground set.
        :return: Whether ``subset`` is an independent set in this matroid.
        """
        pass

    def is_independent_incremental(
        self, independent_subset: tp.AbstractSet[T], new_element: T
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

    class StatefulIndependenceChecker:
        """
        Return type for the :meth:`is_independent_incremental_stateful` method.

        Keeps track of a subset which is assumed to be independent at all times and
        which is modified incrementally (adding one element at a time).

        Subclasses should redefine this nested class (inheriting from this one) in
        order to implement :meth:`is_independent_incremental_stateful`; see that method
        for details.
        """

        def __init__(
            self, matroid: "Matroid", independent_subset: tp.MutableSet[T]
        ):
            self.matroid = matroid
            # current independent subset
            self.independent_subset = independent_subset

        def would_be_independent_after_adding(self, element: T) -> bool:
            """
            Whether the subset would stay independent after adding ``new_element``.

            Doesn't actually add the given element to the current subset; use
            :meth:`add_element` for that.
            """
            # default implementation falls back on simpler methods
            return self.matroid.is_independent_incremental(
                self.independent_subset, element
            )

        def add_element(self, element: T) -> None:
            """
            Add an element to the subset such that the latter remains independent.

            The fact that adding the given element maintains the independence of the
            subset is assumed as a precondition. For instance, one could check it using
            :meth:`would_be_independent_after_adding` first and then only call this
            method if the result was ``True``.
            """
            self.independent_subset.add(element)

        @tp.final
        def add_if_independent(self, element: T) -> bool:
            """
            Shortcut for ``would_be_independent_after_adding`` + ``add_element``.

            Checks if adding the given element would maintain independence; if so, it
            adds it to the subset.

            :returns: Whether the given element was added to the subset, maintaining
                independence.
            """
            if self.would_be_independent_after_adding(element):
                self.add_element(element)
                return True
            else:
                return False

    @tp.final
    def stateful_independence_checker(
        self, independent_subset: tp.MutableSet[T]
    ) -> StatefulIndependenceChecker:
        """
        Special stateful implementation of an incremental independence check.

        This method provides an implementation of the independence check in cases in
        which we want to test a sequence of subsets (S0, S1, ..., Sn, ...) such that
        each successive set is obtained by adding a new element to the previous.

        The interface is as follows. This method takes a (mutable) set S, which is
        assumed to be independent, and returns a ``StatefulIndependenceChecker``
        object that keeps track of S. The initial value of S acts as the "seed" for
        the sequence (S0 above), and is usually the empty set. The caller can use
        the ``.would_be_independent_after_adding(x)`` method of the returned object
        to test whether S + x is independent; if it is, the caller can then add
        x to S by calling ``.add_element(x)`` on the returned object.

        The given set is modified in-place, so the caller will be able to see the
        modifications on the set S. On the other hand, the caller shouldn't modify S
        by other means while the returned object is still being used, otherwise it
        will result in undefined behaviour.

        The purpose of this method is to allow certain subclasses to provide a
        specialized implementation for this special case that is more efficient than
        the direct independence test or successive uses of
        :meth:`is_independent_incremental`. The default implementation falls back on
        :meth:`is_independent_incremental`. To provide such a specialized
        implementation, inheritors must override the nested class
        :class:`Matroid.StatefulIndependenceChecker`, not this method itself.

        In particular, this special case is used in the greedy algorithm for finding
        a maximal independent set.

        :param independent_subset: The subset of elements S which is modified as
            described above. As a precondition, it must be an independent set. Its
            initial value serves as the starting point for the sequence of sets.

        :return: An object of the nested class ``StatefulIncrementalChecker`` satisfying
            the above specification.
        """
        return self.StatefulIndependenceChecker(
            matroid=self, independent_subset=independent_subset
        )

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

    def total_weight(self, subset: tp.AbstractSet[T]) -> float:
        """
        Utility to compute the total weight of a subset of elements.

        :param subset: Subset of element of the ground set. Behaviour is undefined if
            this is not a set (contains duplicates) or is not a subset of the ground
            set of this matroid.
        :return: The sum of the weights of the elements in the given subset.
        """
        return sum(map(self.get_weight, subset))


class MutableMatroid(Matroid[T], metaclass=abc.ABCMeta):
    """Base class for mutable matroid subclasses."""

    @abc.abstractmethod
    def add_element(self, element: T, weight: tp.Optional[float] = None) -> None:
        """
        Add an element to the matroid.

        The element will be added to the ground set. Whether the independent sets are
        modified depends on the concrete subclass.

        If the element is already in the matroid, the ground set will remain the same;
        however if a custom weight is specified, the weight of the element will be
        updated.

        :param element: Element to be added.
        :param weight: Weight of the new element (default is 1.0).
        """
        pass

    @abc.abstractmethod
    def remove_element(self, element: T) -> None:
        """
        Remove an element from the matroid.

        The element will be removed from the ground set and any independent sets
        containing it.

        :param element: Element to be removed.
        """
        pass
