import itertools
from typing import Collection, FrozenSet, Iterable, Iterator, Optional


def generate_subsets(
    s: Collection, sizes: Optional[Iterable[int]] = None
) -> Iterator[FrozenSet]:
    """
    Generate all subsets of a set of the given sizes, in ascending order of size.

    :param s: Set from which to generate subsets. Typed as an arbitrary collection
        for convenience, but as a precondition it should look like a set (i.e. not have
        any duplicates).
    :param sizes: Sizes of the subsets to generate; by default, all possible sizes.
    :return: All subsets of each of the given sizes.
    """
    sizes = sorted(set(sizes)) if sizes is not None else range(len(s) + 1)
    for cardinality in sizes:
        yield from map(frozenset, itertools.combinations(s, r=cardinality))
