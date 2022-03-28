#!/usr/bin/env python3
"""
Runs performance experiments for removing elements under the dynamic algorithms.
"""

import copy
import functools
import random
import typing as tp

import numpy as np

from matroids.algorithms.dynamic import (
    DynamicMaximalIndependentSetAlgorithm,
    NaiveDynamic,
    PartialDynamicMaximalIndependentSetAlgorithm,
    RestartGreedy,
    dynamic_removal_maximal_independent_set_uniform_weights,
)
from utils.generate import (
    generate_random_graphical_matroid,
)
from utils.performance_experiment import (
    InputData,
    PerformanceExperiment,
    PerformanceExperimentGroup,
)
from utils.seed import set_seed
from utils.stopwatch import Stopwatch


set_seed(2022)


def input_generator(
    size: int, rank: int, uniform_weights: bool
) -> tp.Iterator[InputData]:
    while True:
        # reuse same matroid for 10 instances
        matroid = generate_random_graphical_matroid(
            size, rank, uniform_weights=uniform_weights
        )
        elements = list(matroid.ground_set)
        for _ in range(10):
            yield {
                "matroid": matroid,
                "element_to_remove": random.choice(elements),
            }


def time_partial_dynamic(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm,
    matroid,
    element_to_remove,
) -> float:
    """Time one run of the given dynamic algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = copy.deepcopy(matroid)

    # start generator (only want to time dynamic part)
    remover = algorithm(matroid)
    remover.send(None)

    with Stopwatch() as stopwatch:
        remover.send(element_to_remove)

    return stopwatch.measurement


def time_full_dynamic(
    algorithm: DynamicMaximalIndependentSetAlgorithm,
    matroid,
    element_to_remove,
):
    """Time one run of the given dynamic algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = copy.deepcopy(matroid)
    algorithm_instance = algorithm(matroid)

    with Stopwatch() as stopwatch:
        algorithm_instance.remove_element(element_to_remove)

    return stopwatch.measurement


non_uniform_timers = {
    "restart_greedy": functools.partial(time_full_dynamic, RestartGreedy),
    "naive_dynamic": functools.partial(time_full_dynamic, NaiveDynamic),
}


uniform_timers = non_uniform_timers | {
    "uniform_weights_dynamic": functools.partial(
        time_partial_dynamic, dynamic_removal_maximal_independent_set_uniform_weights
    ),
}


deletion_fixed_size_varying_rank = PerformanceExperimentGroup(
    identifier="deletion_fixed_size_varying_rank",
    title="Time per deletion (fixed size, varying rank)",
    experiments=[
        PerformanceExperiment(
            title=f"size = {size}",
            timer_functions=non_uniform_timers,
            x_name="rank",
            x_range=np.linspace(40, size, num=10, dtype=int),
            fixed_variables={"size": size, "uniform_weights": False},
            input_generator=input_generator,
            generated_inputs=100,
        )
        for size in [500]
    ],
)


deletion_fixed_rank_varying_size = PerformanceExperimentGroup(
    identifier="deletion_fixed_rank_varying_size",
    title="Time per deletion (varying size, fixed rank)",
    experiments=[
        PerformanceExperiment(
            title=f"rank = {rank}",
            timer_functions=non_uniform_timers,
            x_name="size",
            x_range=np.linspace(100, 500, num=10, dtype=int),
            fixed_variables={"rank": rank, "uniform_weights": False},
            input_generator=input_generator,
            generated_inputs=100,
        )
        for rank in [100]
    ],
)


uniform_deletion_fixed_size_varying_rank = PerformanceExperimentGroup(
    identifier="uniform_deletion_fixed_size_varying_rank",
    title="Time per deletion (uniform weights, fixed size, varying rank)",
    experiments=[
        PerformanceExperiment(
            title=f"size = {size}",
            timer_functions=uniform_timers,
            x_name="rank",
            x_range=np.linspace(20, size, num=10, dtype=int),
            fixed_variables={"size": size, "uniform_weights": True},
            input_generator=input_generator,
            generated_inputs=100,
        )
        for size in [150]
    ],
)

uniform_deletion_fixed_rank_varying_size = PerformanceExperimentGroup(
    identifier="uniform_deletion_fixed_rank_varying_size",
    title="Time per deletion (uniform weights, varying size, fixed rank)",
    experiments=[
        PerformanceExperiment(
            title=f"rank = {rank}",
            timer_functions=uniform_timers,
            x_name="size",
            x_range=np.linspace(100, 400, num=10, dtype=int),
            fixed_variables={"rank": rank, "uniform_weights": True},
            input_generator=input_generator,
            generated_inputs=100,
        )
        for rank in [40, 80]
    ],
)


deletion_fixed_size_varying_rank.measure_show_and_save(
    plot_kind="mean&range",
    legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, 0.0)},
)
deletion_fixed_rank_varying_size.measure_show_and_save(
    plot_kind="mean&range",
    legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, 0.0)},
)
uniform_deletion_fixed_size_varying_rank.measure_show_and_save()
uniform_deletion_fixed_rank_varying_size.measure_show_and_save(
    legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, 0.0)},
)
