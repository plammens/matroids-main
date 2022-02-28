"""Algorithms for dynamic MIS that handle both addition and removal of elements."""

import abc
import typing as tp

import llist

from matroids.matroid import MutableMatroid, T
from matroids.utils.linked_list_set import LinkedListSet, iter_nodes
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
    def add_element(
        self, element: T, /, weight: tp.Optional[float] = None
    ) -> tp.Set[T]:
        """
        Add an element to the matroid and return the new maximal independent set.

        If the element is already in the matroid, its weight is updated to the specified
        value, if given.

        :param element: Element to add.
        :param weight: Optional weight for the new element.

        :return: The new maximal independent set after adding the given element.
        """
        pass

    @abc.abstractmethod
    def remove_element(self, element, /) -> tp.Set[T]:
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

    def add_element(
        self, element: T, /, weight: tp.Optional[float] = None
    ) -> tp.Set[T]:
        self._matroid.add_element(element, weight=weight)
        self._current = result = maximal_independent_set(self._matroid)
        return result

    def remove_element(self, element: T, /) -> tp.Set[T]:
        self._matroid.remove_element(element)
        self._current = result = maximal_independent_set(self._matroid)
        return result


class NaiveDynamic(DynamicMaximalIndependentSetAlgorithm):
    def __init__(self, matroid: MutableMatroid[T]):
        super().__init__(matroid)

        # linked list of elements with non-negative weight in descending order of weight
        elements = filter(lambda x: matroid.get_weight(x) >= 0, matroid.ground_set)
        elements = sorted(elements, key=matroid.get_weight, reverse=True)
        self._elements: LinkedListSet[T] = LinkedListSet(elements)

        # parallel list of indicators (whether each element was added to the MIS)
        self._indicators: llist.dllist = llist.dllist([False] * len(self._elements))

        # use greedy for initial solution
        independence_checker = matroid.stateful_independence_checker(set())
        self._current_solution = self._continue_greedy(
            independence_checker,
            self._elements.first,
            self._indicators.first,
        )

    @property
    def current(self) -> tp.Set[T]:
        return self._current_solution

    def add_element(
        self, new_element: T, /, weight: tp.Optional[float] = None
    ) -> tp.Set[T]:
        get_weight = self._matroid.get_weight  # for efficiency

        # shortcuts when the element is already in the matroid
        if new_element in self._matroid.ground_set:
            if weight is None or get_weight(new_element) == weight:
                # element already in matroid and weight doesn't change
                return self._current_solution
            else:
                # element already present but weight changes
                # easiest is to delete and then add
                self.remove_element(new_element)
                return self.add_element(new_element, weight=weight)

        self._matroid.add_element(new_element, weight)
        weight: float = get_weight(new_element)  # actual weight after adding
        if weight < 0:
            # new element with negative weight; solution doesn't change
            return self._current_solution

        # insert element in sorted position; meanwhile reconstruct independent set
        independence_checker, element_node, indicator_node = self._reconstruct_greedy(
            until=lambda e: get_weight(e) < weight
        )
        element_node = self._elements.insert(position=element_node, value=new_element)
        indicator_node = self._indicators.insert(False, indicator_node)

        # possible shortcut
        if not independence_checker.would_be_independent_after_adding(new_element):
            return self._current_solution
        else:
            independence_checker.add_element(new_element)
            indicator_node.value = True
            # already dealt with new_element as a special case; continue with the next
            return self._continue_greedy(
                independence_checker,
                elements_start=element_node.next,
                indicators_start=indicator_node.next,
            )

    def remove_element(self, element_to_remove, /) -> tp.Set[T]:
        self._matroid.remove_element(element_to_remove)

        # shortcut if element is not a pivot
        if element_to_remove not in self._current_solution:
            self._elements.discard(element_to_remove)
            return self._current_solution

        # find sorted position of deleted element; meanwhile reconstruct independent set
        independence_checker, element_node, indicator_node = self._reconstruct_greedy(
            until=lambda e: e == element_to_remove
        )
        indicator_node_to_remove = indicator_node
        element_node, indicator_node = element_node.next, indicator_node.next
        self._elements.remove(element_to_remove)
        self._indicators.remove(indicator_node_to_remove)

        return self._continue_greedy(
            independence_checker,
            elements_start=element_node,
            indicators_start=indicator_node,
        )

    def _reconstruct_greedy(
        self, until: tp.Callable[[T], bool]
    ) -> tp.Tuple[
        MutableMatroid.StatefulIndependenceChecker, llist.dllistnode, llist.dllistnode
    ]:
        """
        Reconstruct the state of the greedy algorithm up to a certain point.

        :param until: Reconstruct until this predicate becomes true for the
            current element.

        :returns: The independence checker object and the list position as they were in
            the greedy algorithm at the step in which ``until_predicate`` becomes true.
        """
        independent_set: tp.Set[T] = set()
        checker = self._matroid.stateful_independence_checker(independent_set)

        # (this search is linear, but it doesn't matter since we need to re-run
        # part of the greedy algorithm afterwards, which is also linear)
        for element_node, indicator_node in zip(
            self._elements.iter_nodes(), self._indicators.iternodes()
        ):
            element = element_node.value
            if until(element):
                break
            if indicator_node.value:
                checker.add_element(element)
        else:
            # in case there are no elements or position is the end of the list
            element_node, indicator_node = None, None

        return checker, element_node, indicator_node

    def _continue_greedy(
        self,
        independence_checker: MutableMatroid.StatefulIndependenceChecker,
        elements_start: llist.dllistnode,
        indicators_start: llist.dllistnode,
    ) -> tp.Set[T]:
        """Run the greedy algorithm from the given element onwards."""
        for element, indicator_node in zip(
            self._elements.iter_values(start=elements_start),
            iter_nodes(indicators_start),
        ):
            indicator_node.value = independence_checker.add_if_independent(element)

        self._current_solution = solution = independence_checker.independent_subset
        return solution  # noqa
