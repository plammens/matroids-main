#!/usr/bin/env python3
"""
Runs performance experiments for adding elements under the dynamic algorithms.
"""


import copy
import functools
import random
import typing as tp

import numpy as np

from matroids.algorithms.dynamic import (
    DynamicMaximalIndependentSetAlgorithm,
    NaiveDynamic,
    RestartGreedy,
)
from utils.generate import (
    generate_random_graphical_matroid,
)
from utils.misc import compute_missing_edges
from utils.performance_experiment import (
    InputData,
    PerformanceExperiment,
    PerformanceExperimentGroup,
)
from utils.seed import set_seed
from utils.stopwatch import Stopwatch


set_seed(2022)


def input_generator(size: int, rank: int) -> tp.Iterator[InputData]:
    while True:
        # reuse same matroid for 10 instances
        matroid = generate_random_graphical_matroid(size, rank, uniform_weights=False)
        missing_edges = list(compute_missing_edges(matroid.graph, extra_nodes={-1}))
        for _ in range(10):
            yield {
                "matroid": matroid,
                "element_to_add": random.choice(missing_edges),
                "weight": np.random.uniform(low=-1.0, high=1.0),
            }


def time_full_dynamic(
    algorithm: DynamicMaximalIndependentSetAlgorithm,
    matroid,
    element_to_add,
    weight: float,
):
    """Time one run of the given dynamic algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = copy.deepcopy(matroid)
    algorithm_instance = algorithm(matroid)

    with Stopwatch() as stopwatch:
        algorithm_instance.add_element(element_to_add, weight=weight)

    return stopwatch.measurement


timers = {
    "restart_greedy": functools.partial(time_full_dynamic, RestartGreedy),
    "naive_dynamic": functools.partial(time_full_dynamic, NaiveDynamic),
}


addition_fixed_size_varying_rank = PerformanceExperimentGroup(
    identifier="addition_fixed_size_varying_rank",
    title="Time per addition (fixed size, varying rank)",
    experiments=[
        PerformanceExperiment(
            title=f"size = {size}",
            timer_functions=timers,
            x_name="rank",
            x_range=np.linspace(40, size, num=10, dtype=int),
            fixed_variables={"size": size},
            input_generator=input_generator,
            generated_inputs=100,
        )
        for size in [500]
    ],
)


addition_fixed_rank_varying_size = PerformanceExperimentGroup(
    identifier="addition_fixed_rank_varying_size",
    title="Time per addition (varying size, fixed rank)",
    experiments=[
        PerformanceExperiment(
            title=f"rank = {rank}",
            timer_functions=timers,
            x_name="size",
            x_range=np.linspace(100, 500, num=10, dtype=int),
            fixed_variables={"rank": rank},
            input_generator=input_generator,
            generated_inputs=100,
        )
        for rank in [50]
    ],
)


addition_fixed_size_varying_rank.measure_show_and_save(
    plot_kind="mean&range",
    legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, 0.0)},
)
addition_fixed_rank_varying_size.measure_show_and_save(
    plot_kind="mean&range",
    legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, 0.0)},
)
