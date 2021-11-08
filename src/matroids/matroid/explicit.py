import dataclasses
import typing

from .base import MutableMatroid
from matroids.utils import generate_subsets


@dataclasses.dataclass(frozen=True)
class ExplicitMatroid(MutableMatroid[typing.Any]):
    """
    A matroid based on an explicit ground set and independent sets family.

    Care must be taken so that the given tuple of (elements, independent_sets) does
    indeed satisfy the axioms of a matroid.
    """

    elements: typing.Set[typing.Any]  #: the explicit ground set
    independent_sets: typing.Set[typing.FrozenSet[typing.Any]]
    weights: typing.MutableMapping[
        typing.Any, float
    ] = None  #: mapping of elements to weights

    def __post_init__(self):
        # initialise default weights
        if self.weights is None:
            weights = {e: 1.0 for e in self.ground_set}
            object.__setattr__(self, "weights", weights)

    __hash__ = None

    T = typing.TypeVar("T")

    @classmethod
    def uniform(
        cls,
        elements: typing.Collection[T],
        k: int,
        weights: typing.Optional[typing.MutableMapping[T, float]] = None,
    ) -> "ExplicitMatroid":
        """
        Make a uniform matroid: where every set of size at most k is independent.

        :param elements: Elements making up the ground set.
        :param k: The independent sets will be those of cardinality <= ``k``.
        :param weights: Optionally specify custom weights for each element.
        :return: A new uniform matroid whose ground set is ``frozenset(elements)``
            and whose independent sets are all the subsets of the ground set that
            are of cardinality <= ``k``.
        """
        ground_set = set(elements)
        independent_sets = set(generate_subsets(ground_set, sizes=range(k + 1)))
        if weights is not None and set(weights) != ground_set:
            raise ValueError("Keys of weights mapping don't coincide with elements.")
        return cls(ground_set, independent_sets, weights)

    del T

    @property
    def ground_set(self) -> typing.FrozenSet[typing.Any]:
        return frozenset(self.elements)

    def is_independent(self, subset: typing.AbstractSet[typing.Any]) -> bool:
        return frozenset(subset) in self.independent_sets

    def get_weight(self, element: typing.Any) -> float:
        return self.weights[element]

    def add_element(self, element) -> None:
        self.elements.add(element)
        self.weights[element] = 1.0

    def remove_element(self, element) -> None:
        self.elements.remove(element)
        del self.weights[element]
        to_change = []
        for independent_set in self.independent_sets:
            if element in independent_set:
                to_change.append((independent_set, independent_set - {element}))
        for before, after in to_change:
            self.independent_sets.remove(before)
            self.independent_sets.add(after)
