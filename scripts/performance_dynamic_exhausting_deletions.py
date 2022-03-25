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
)
from utils.generate import generate_random_dummy_matroid
from utils.seed import set_seed
from utils.performance_experiment import (
    InputData,
    PerformanceExperiment,
    PerformanceExperimentGroup,
)
from utils.stopwatch import Stopwatch


set_seed(2022)


def input_generator(size: int, rank: int) -> tp.Iterator[InputData]:
    matroid = generate_random_dummy_matroid(size, rank, uniform_weights=True)

    elements = list(matroid.ground_set)
    while True:
        random.shuffle(elements)
        yield {"matroid": matroid, "removal_sequence": elements}


def time_restart_greedy(
    matroid: MutableIntUniformMatroid,
    removal_sequence: tp.Sequence[int],
) -> float:
    """Time one run of the greedy algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = dataclasses.replace(matroid)

    with Stopwatch() as stopwatch:
        for element in removal_sequence:
            matroid.remove_element(element)
            maximal_independent_set(matroid)

    return stopwatch.measurement


def time_partial_dynamic(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm,
    matroid: MutableIntUniformMatroid,
    removal_sequence: tp.Sequence[int],
) -> float:
    """Time one run of the given dynamic algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = dataclasses.replace(matroid)

    # start generator (only want to time dynamic part)
    remover = algorithm(matroid)
    remover.send(None)

    with Stopwatch() as stopwatch:
        for element in removal_sequence:
            remover.send(element)

    return stopwatch.measurement


def time_full_dynamic(
    algorithm: DynamicMaximalIndependentSetAlgorithm,
    matroid: MutableIntUniformMatroid,
    removal_sequence: tp.Sequence[int],
):
    """Time one run of the given dynamic algorithm; return time in seconds."""
    # make copy of shared matroid (because it's mutable)
    matroid = dataclasses.replace(matroid)
    algorithm_instance = algorithm(matroid)

    with Stopwatch() as stopwatch:
        for element in removal_sequence:
            algorithm_instance.remove_element(element)

    return stopwatch.measurement


timers = {
    "restart_greedy": time_restart_greedy,
    "naive_dynamic": functools.partial(time_full_dynamic, NaiveDynamic),
    "uniform_weights_dynamic": functools.partial(
        time_partial_dynamic, dynamic_removal_maximal_independent_set_uniform_weights
    ),
}

# versions that divide total time by number of removals
timers_per_removal = {
    label: lambda _timer=timer, **kwargs: _timer(**kwargs) / kwargs["matroid"].size
    for label, timer in timers.items()
}


size_experiments = PerformanceExperimentGroup(
    identifier="exhausting_deletions_fixed_size_varying_rank",
    title="Total time over exhausting sequence of deletions\n"
    "uniform weights, fixed size, varying rank",
    experiments=[
        PerformanceExperiment(
            title=f"size = {size}",
            timer_functions=timers,
            x_name="rank",
            x_range=np.linspace(0, size, num=10, dtype=int),
            fixed_variables={"size": size},
            input_generator=input_generator,
            generated_inputs=3,
        )
        for size in [200]
    ],
)

rank_experiments = PerformanceExperimentGroup(
    identifier="exhausting_deletions_normalised_fixed_rank_varying_size",
    title="Time per deletion over exhausting sequence of deletions\n"
    f"uniform weights, varying size, fixed rank",
    experiments=[
        PerformanceExperiment(
            title=f"rank = {rank}",
            timer_functions=timers_per_removal,
            x_name="size",
            x_range=np.linspace(100, 500, num=10, dtype=int),
            fixed_variables={"rank": rank},
            input_generator=input_generator,
            generated_inputs=3,
        )
        for rank in [5, 10, 15]
    ],
)


size_experiments.measure_show_and_save()
rank_experiments.measure_show_and_save()
