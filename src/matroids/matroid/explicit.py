import dataclasses
import typing

from matroids.matroid import Matroid
from matroids.utils import generate_subsets


@dataclasses.dataclass(frozen=True)
class ExplicitMatroid(Matroid[typing.Any]):
    """
    A matroid based on an explicit ground set and independent sets family.

    Care must be taken so that the given tuple of (elements, independent_sets) does
    indeed satisfy the axioms of a matroid.
    """

    elements: typing.FrozenSet[typing.Any]  #: the explicit ground set
    independent_sets: typing.FrozenSet[typing.FrozenSet[typing.Any]]
    weights: typing.Mapping[typing.Any, float] = None  #: mapping of elements to weights

    def __post_init__(self):
        # initialise default weights
        if self.weights is None:
            weights = {e: 1.0 for e in self.ground_set}
            object.__setattr__(self, "weights", weights)

    T = typing.TypeVar("T")

    @classmethod
    def uniform(
        cls,
        elements: typing.Collection[T],
        k: int,
        weights: typing.Optional[typing.Mapping[T, float]] = None,
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
        ground_set = frozenset(elements)
        independent_sets = frozenset(generate_subsets(ground_set, sizes=range(k + 1)))
        if weights is not None and frozenset(weights) != ground_set:
            raise ValueError("Keys of weights mapping don't coincide with elements.")
        return cls(ground_set, independent_sets, weights)  # noqa

    del T

    @property
    def ground_set(self) -> typing.Set[typing.Any]:
        return set(self.elements)

    def is_independent(self, subset: typing.Set[typing.Any]) -> bool:
        return frozenset(subset) in self.independent_sets

    def get_weight(self, element: typing.Any) -> float:
        return self.weights[element]
