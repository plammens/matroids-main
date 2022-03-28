import random
import typing as tp

import pytest

from matroids.utils import LinkedListSet, RandomAccessMutableSet


SET_CLASSES_TO_TEST = [
    RandomAccessMutableSet,
    LinkedListSet,
]

ELEMENTS = [
    (1, 2, 2, 3, 1),
    "abc",
    (None, 3, "x", None),
]


@pytest.mark.parametrize("elements", ELEMENTS)
@pytest.mark.parametrize("set_class", SET_CLASSES_TO_TEST)
def test_setConstruction_correct(
    set_class: tp.Type[tp.MutableSet], elements: tp.Iterable
):
    assert set_class(elements) == set(elements)  # noqa


@pytest.mark.parametrize("elements", ELEMENTS)
def test_randomAccessSet_randomChoice_works(elements: tp.Iterable):
    random_access_set = RandomAccessMutableSet(elements)
    choice = random.choice(random_access_set)
    assert choice in elements
    assert random_access_set == set(elements)  # shouldn't have changed
