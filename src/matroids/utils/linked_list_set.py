import operator
import typing as tp

import llist


T = tp.TypeVar("T")


class LinkedListSet(tp.Collection[T]):
    """
    Mutable data structure consisting of a linked list of unique elements.

    Supports:
     - O(1) deletion by value
     - O(1) insertion given a position (node) and a value
    """

    def __init__(self, elements: tp.Iterable[T]):
        self._llist = llist.dllist(elements)
        self._value_to_node: tp.Dict[T, llist.dllistnode] = {
            node.value: node for node in self._llist.iternodes()
        }

    def __repr__(self):
        return f"{type(self).__name__}({list(self)!r})"

    def __len__(self) -> int:
        return len(self._llist)

    def __iter__(self) -> tp.Iterator[T]:
        return iter(self._llist)

    def __contains__(self, x: object) -> bool:
        return x in self._value_to_node

    @property
    def first(self) -> llist.dllistnode:
        return self._llist.first

    @property
    def last(self) -> llist.dllistnode:
        return self._llist.last

    def iter_nodes(
        self, start: tp.Optional[llist.dllistnode] = None
    ) -> tp.Iterator[llist.dllistnode]:
        """
        Iterate over the nodes of the linked list.

        :param start: If given, start iterating from this node (otherwise iterate
            from the beginning of the list).
        :return: An iterator over the nodes starting at the specified position.
        """
        if start is None:
            return self._llist.iternodes()  # optimized C code
        else:
            return iter_nodes(start)

    def iter_values(self, start: tp.Optional[llist.dllistnode] = None) -> tp.Iterator[T]:
        """
        Iterate over the nodes of the linked list.

        :param start: If given, start iterating from this node (otherwise iterate
            from the beginning of the list).
        :return: An iterator over the values starting at the specified position.
        """
        return map(operator.attrgetter("value"), self.iter_nodes(start=start))

    def insert(
        self, position: tp.Optional[llist.dllistnode], value: T
    ) -> llist.dllistnode:
        """
        Insert a value if it doesn't already exist.

        :param position: The value, if not already present, will be inserted immediately
            before this node. If ``position`` is None, it will be inserted at the end.
        :param value: Value to insert.

        :return: The node (regardless of whether it is new or it already existed)
            containing the given value.
        """
        if value not in self._value_to_node:
            node = self._llist.insert(value, position)
            self._value_to_node[value] = node
            return node
        else:
            return self._value_to_node[value]

    def remove(self, value: T):
        """Remove the given value; raise KeyError if not in the list."""
        node = self._value_to_node.pop(value)
        self._llist.remove(node)

    def find(self, value: T) -> llist.dllistnode:
        """Get the node corresponding to the given value."""
        return self._value_to_node[value]


def iter_nodes(start: llist.dllistnode) -> tp.Iterator[llist.dllistnode]:
    """Iterate from the given linked-list node onward."""
    node = start
    while node is not None:
        yield node
        node = node.next
