#!/usr/bin/env python3
"""
Performance check for matroids.utils.random_access_set.RandomAccessMutableSet.
"""

import functools
import random
import typing as tp

import numpy as np

from matroids.utils.random_access_set import RandomAccessMutableSet
from utils.performance_experiment import (
    InputData,
    PerformanceExperiment,
    PerformanceExperimentGroup,
)
from utils.seed import set_seed
from utils.stopwatch import Stopwatch


set_seed(2022)


def input_generator(size: int) -> tp.Iterator[InputData]:
    while True:
        yield {"size": size, "element_to_remove": random.randrange(size)}


def time_set_remove(
    set_class: tp.Type[tp.MutableSet], size: int, element_to_remove: int
) -> float:
    set_instance = set_class(range(size))  # noqa

    with Stopwatch() as stopwatch:
        set_instance.remove(element_to_remove)

    return stopwatch.measurement


timers = {
    "builtins.set": functools.partial(time_set_remove, set),
    "RandomAccessMutableSet": functools.partial(
        time_set_remove, RandomAccessMutableSet
    ),
}


PerformanceExperimentGroup(
    identifier="random_access_set_remove",
    title="Time for removing a random element from the set",
    experiments=[
        PerformanceExperiment(
            timer_functions=timers,
            x_name="size",
            x_range=np.linspace(100, 8_000, num=10, dtype=int),
            input_generator=input_generator,
            generated_inputs=100,
        )
    ],
).measure_show_and_save()
