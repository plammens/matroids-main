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
from matroids.matroid import MutableMatroid, T
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


def input_generator(size: int, rank: int) -> tp.Iterator[InputData]:
    while True:
        # reuse same matroid for 3 instances
        matroid = generate_random_graphical_matroid(size, rank, uniform_weights=True)
        elements = list(matroid.ground_set)
        for _ in range(3):
            random.shuffle(elements)
            yield {"matroid": matroid, "removal_sequence": elements}


def time_partial_dynamic(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm,
    matroid: MutableMatroid[T],
    removal_sequence: tp.Sequence[T],
) -> float:
    """Time one run of the given dynamic algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = copy.deepcopy(matroid)

    # start generator (only want to time dynamic part)
    remover = algorithm(matroid)
    remover.send(None)

    with Stopwatch() as stopwatch:
        for element in removal_sequence:
            remover.send(element)

    return stopwatch.measurement


def time_full_dynamic(
    algorithm: DynamicMaximalIndependentSetAlgorithm,
    matroid: MutableMatroid[T],
    removal_sequence: tp.Sequence[T],
):
    """Time one run of the given dynamic algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = copy.deepcopy(matroid)
    algorithm_instance = algorithm(matroid)

    with Stopwatch() as stopwatch:
        for element in removal_sequence:
            algorithm_instance.remove_element(element)

    return stopwatch.measurement


timers = {
    "restart_greedy": functools.partial(time_full_dynamic, RestartGreedy),
    "naive_dynamic": functools.partial(time_full_dynamic, NaiveDynamic),
    "uniform_weights_dynamic": functools.partial(
        time_partial_dynamic, dynamic_removal_maximal_independent_set_uniform_weights
    ),
}

# versions that divide total time by number of removals
timers_per_removal = {
    label: lambda _timer=timer, **variables: (
        _timer(**variables) / len(variables["matroid"].ground_set)
    )
    for label, timer in timers.items()
}


rank_experiments = PerformanceExperimentGroup(
    identifier="exhausting_deletions_fixed_size_varying_rank",
    title="Total time over exhausting sequence of deletions\n"
    "uniform weights, fixed size, varying rank",
    experiments=[
        PerformanceExperiment(
            title=f"size = {size}",
            timer_functions=timers,
            x_name="rank",
            x_range=np.linspace(25, size, num=10, dtype=int),
            fixed_variables={"size": size},
            input_generator=input_generator,
            generated_inputs=15,
            repeats=3,
        )
        for size in [200]
    ],
)

size_experiments = PerformanceExperimentGroup(
    identifier="exhausting_deletions_normalised_fixed_rank_varying_size",
    title="Time per deletion over exhausting sequence of deletions\n"
    f"uniform weights, varying size, fixed rank",
    experiments=[
        PerformanceExperiment(
            title=f"rank = {rank}",
            timer_functions=timers_per_removal,
            x_name="size",
            x_range=np.linspace(100, 400, num=10, dtype=int),
            fixed_variables={"rank": rank},
            input_generator=input_generator,
            generated_inputs=15,
            repeats=3,
        )
        for rank in [30, 45, 60]
    ],
)


rank_experiments.measure_show_and_save()
size_experiments.measure_show_and_save(
    legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, 0.0)}
)
