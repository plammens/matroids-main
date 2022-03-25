import dataclasses
import functools
import random
import typing as tp

import numpy as np

from matroids.algorithms.dynamic import (
    DynamicMaximalIndependentSetAlgorithm,
    NaiveDynamic,
    PartialDynamicMaximalIndependentSetAlgorithm,
    dynamic_removal_maximal_independent_set_uniform_weights,
)
from matroids.algorithms.static import maximal_independent_set
from matroids.matroid import (
    MutableIntUniformMatroid,
    MutableMatroid,
)
from utils.performance_experiment import (
    InputData,
    PerformanceExperiment,
    PerformanceExperimentGroup,
)
from utils.stopwatch import Stopwatch


random.seed(2022)


def generate_dummy_matroid(
    size: int, rank: int, uniform_weights: bool
) -> MutableMatroid:
    weights = {} if uniform_weights else {i: random.random() for i in range(size)}
    return MutableIntUniformMatroid(size, rank, weights)


def input_generator(
    size: int, rank: int, uniform_weights: bool
) -> tp.Iterator[InputData]:
    matroid = generate_dummy_matroid(size, rank, uniform_weights)

    elements = list(matroid.ground_set)
    random.shuffle(elements)

    for element in elements:
        yield {"matroid": matroid, "element_to_remove": element}


def time_restart_greedy(
    matroid: MutableIntUniformMatroid,
    element_to_remove: int,
) -> float:
    """Time one run of the greedy algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = dataclasses.replace(matroid)

    with Stopwatch() as stopwatch:
        matroid.remove_element(element_to_remove)
        maximal_independent_set(matroid)

    return stopwatch.measurement


def time_partial_dynamic(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm,
    matroid: MutableIntUniformMatroid,
    element_to_remove: int,
) -> float:
    """Time one run of the given dynamic algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = dataclasses.replace(matroid)

    # start generator (only want to time dynamic part)
    remover = algorithm(matroid)
    remover.send(None)

    with Stopwatch() as stopwatch:
        remover.send(element_to_remove)

    return stopwatch.measurement


def time_full_dynamic(
    algorithm: DynamicMaximalIndependentSetAlgorithm,
    matroid: MutableIntUniformMatroid,
    element_to_remove: int,
):
    """Time one run of the given dynamic algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = dataclasses.replace(matroid)
    algorithm_instance = algorithm(matroid)

    with Stopwatch() as stopwatch:
        algorithm_instance.remove_element(element_to_remove)

    return stopwatch.measurement


timers = {
    "restart_greedy": time_restart_greedy,
    "naive_dynamic": functools.partial(time_full_dynamic, NaiveDynamic),
    "uniform_weights_dynamic": functools.partial(
        time_partial_dynamic, dynamic_removal_maximal_independent_set_uniform_weights
    ),
}


size_experiments = PerformanceExperimentGroup(
    identifier="fixed_size_varying_rank",
    title="Time per deletion (uniform weights, fixed size, varying rank)",
    experiments=[
        PerformanceExperiment(
            title=f"size = {size}",
            timer_functions=timers,
            x_name="rank",
            x_range=np.linspace(0, size, num=10, dtype=int),
            fixed_variables={"size": size, "uniform_weights": True},
            input_generator=input_generator,
            generated_inputs=50,
        )
        for size in [150]
    ],
)

rank_experiments = PerformanceExperimentGroup(
    identifier="fixed_rank_varying_size",
    title="Time per deletion (uniform weights, varying size, fixed rank)",
    experiments=[
        PerformanceExperiment(
            title=f"rank = {rank}",
            timer_functions=timers,
            x_name="size",
            x_range=np.linspace(100, 500, num=10, dtype=int),
            fixed_variables={"rank": rank, "uniform_weights": True},
            input_generator=input_generator,
            generated_inputs=50,
        )
        for rank in [25, 50, 100]
    ],
)


size_experiments.measure_show_and_save()
rank_experiments.measure_show_and_save()
