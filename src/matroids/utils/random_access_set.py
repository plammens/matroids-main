import collections.abc
import typing as tp


T = tp.TypeVar("T", covariant=True)


class RandomAccessMutableSet(
    collections.abc.MutableSet[T], collections.abc.Sequence[T]
):
    """
    Data structure providing mutable set operations plus fast random access.

    The random access is only intended for being able to select an element at random
    from the set in O(1) time. This class implements the Sequence interface for
    compatibility with :function:`random.choice`.
    """

    def __init__(self, iterable):
        list_ = []
        element_to_index = {}
        for x in iterable:
            if x not in element_to_index:
                element_to_index[x] = len(list_)
                list_.append(x)

        self._list: tp.List[T] = list_
        self._element_to_index: tp.Dict[T, int] = element_to_index

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({set(self)})"

    def __contains__(self, x: object) -> bool:
        return x in self._element_to_index

    def __len__(self) -> int:
        assert len(self._list) == len(self._element_to_index)
        return len(self._element_to_index)

    def __iter__(self) -> tp.Iterator[T]:
        return iter(self._list)

    def __getitem__(self, i) -> T:
        return self._list[i]

    def add(self, value: T) -> None:
        if value not in self._element_to_index:
            index = len(self)
            self._list.append(value)
            self._element_to_index[value] = index

    def discard(self, value: T) -> None:
        index = self._element_to_index.get(value)
        if index is not None:
            # move last element into the newly created void
            self._list[index] = last = self._list[-1]
            self._list.pop()

            # update element to index mapping
            self._element_to_index[last] = index
            del self._element_to_index[value]
