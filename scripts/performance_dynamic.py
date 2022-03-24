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
    EdgeType,
    MutableIntUniformMatroid,
    MutableMatroid,
)
from utils.performance_plot import PerformanceExperiment
from utils.stopwatch import Stopwatch


rng = np.random.default_rng(seed=2022)


def generate_dummy_matroid(
    size: int, rank: int, uniform_weights: bool = False
) -> MutableMatroid:
    weights = {} if uniform_weights else {i: random.random() for i in range(size)}
    return MutableIntUniformMatroid(size, rank, weights)


def setup(
    size: int, rank: int, *, uniform_weights: bool = True
) -> tp.Tuple[MutableMatroid, tp.List[EdgeType]]:
    matroid = generate_dummy_matroid(size, rank, uniform_weights)

    # select the sequence of removals (over all elements of the matroid)
    removal_sequence = list(matroid.ground_set)
    rng.shuffle(removal_sequence)

    return matroid, removal_sequence


def time_restart_greedy(*args, **kwargs) -> float:
    """Time one run of the greedy algorithm; return time in nanoseconds."""
    matroid, removal_sequence = setup(*args, **kwargs)

    with Stopwatch() as stopwatch:
        for element in removal_sequence:
            matroid.remove_element(element)
            maximal_independent_set(matroid)

    return stopwatch.measurement


def time_partial_dynamic(
    algorithm: PartialDynamicMaximalIndependentSetAlgorithm, *args, **kwargs
) -> float:
    """Time one run of the given dynamic algorithm; return time in seconds."""
    matroid, removal_sequence = setup(*args, **kwargs)

    # start generator (only want to time dynamic part)
    remover = algorithm(matroid)
    remover.send(None)

    with Stopwatch() as stopwatch:
        for element in removal_sequence:
            remover.send(element)

    return stopwatch.measurement


def time_full_dynamic(
    algorithm: DynamicMaximalIndependentSetAlgorithm, *args, **kwargs
):
    """Time one run of the given dynamic algorithm; return time in seconds."""
    matroid, removal_sequence = setup(*args, **kwargs)
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
    label: lambda _timer=timer, **kwargs: _timer(**kwargs) / kwargs["size"]
    for label, timer in timers.items()
}


experiments = [
    PerformanceExperiment(
        title="Total time over exhausting sequence of removals\n"
        f"uniform weights, fixed size ({size}), varying rank",
        timer_functions=timers,
        x_name="rank",
        x_range=np.linspace(0, size, num=10, dtype=int),
        fixed_variables={"size": size},
    )
    for size in [150]
] + [
    PerformanceExperiment(
        title="Time per deletion over exhausting sequence of removals\n"
        f"uniform weights, varying size, fixed rank ({rank})",
        timer_functions=timers_per_removal,
        x_name="size",
        x_range=np.linspace(100, 500, num=10, dtype=int),
        fixed_variables={"rank": rank},
    )
    for rank in [5, 10, 15]
]


plt.style.use(matplotx.styles.dufte)
for experiment in experiments:
    experiment.measure_and_show()
