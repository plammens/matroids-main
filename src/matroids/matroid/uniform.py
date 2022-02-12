import dataclasses
import functools
import typing as tp

from .base import Matroid, MutableMatroid


@dataclasses.dataclass(eq=False)
class IntUniformMatroid(Matroid[int]):
    """
    Dummy uniform matroid made up of integer elements.

    - ``size``: Size of the ground set. The elements of the matroid will be
        {0, ..., size - 1}.
    - ``rank``: Rank of the matroid. A set will be independent in the matroid iff it
        is of size <= ``rank``.
    - ``weights``: Dictionary of overridden weights (default weight is 1.0).

    Used mainly for testing / benchmarking purposes.
    """

    size: int
    rank: int
    weights: tp.Dict[int, float] = dataclasses.field(default_factory=dict)

    @classmethod
    def free(cls, size: int) -> "IntUniformMatroid":
        """Construct an free matroid (rank = size) of the given size."""
        return cls(size=size, rank=size)

    @property
    @functools.cache  # the ground set cannot be reassigned nor mutated, so we can cache
    def ground_set(self) -> tp.AbstractSet[int]:
        return set(range(self.size))

    def is_independent(self, subset: tp.AbstractSet[int]) -> bool:
        return len(subset) <= self.rank

    def get_weight(self, element: int) -> float:
        return self.weights.get(element, 1.0)


@dataclasses.dataclass(eq=False, repr=False)
class MutableIntUniformMatroid(IntUniformMatroid, MutableMatroid[int]):
    """
    Mutable version of :class:`IntUniformMatroid`.

    When an element is added or removed, the rank of the matroid is maintained (i.e.
    the parameter k such that the independent sets are all the sets of size <= k isn't
    modified).
    """

    def __post_init__(self):
        self._elements: tp.Set[int] = set(range(self.size))

    @property
    def ground_set(self) -> tp.AbstractSet[int]:
        return self._elements

    def add_element(self, element: int) -> None:
        self._elements.add(element)

    def remove_element(self, element: int) -> None:
        self._elements.remove(element)
