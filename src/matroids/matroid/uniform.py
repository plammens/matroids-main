import dataclasses
import functools
import typing as tp

from .base import Matroid, T


@dataclasses.dataclass(eq=False)
class IntUniformMatroid(Matroid):
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
    def ground_set(self) -> tp.AbstractSet[T]:
        return set(range(self.size))

    def is_independent(self, subset: tp.AbstractSet[T]) -> bool:
        return len(subset) <= self.rank

    def get_weight(self, element: T) -> float:
        return self.weights.get(element, 1.0)
