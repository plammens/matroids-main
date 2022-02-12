import dataclasses
import functools
import typing as tp

import matplotlib.pyplot as plt
import matplotx
import numpy as np
import tqdm

from matroids.algorithms.dynamic import (
    DynamicMaximalIndependentSetAlgorithm,
    dynamic_removal_maximal_independent_set,
    dynamic_removal_maximal_independent_set_uniform_weights,
)
from matroids.algorithms.static import maximal_independent_set
from matroids.matroid import (
    EdgeType,
    MutableIntUniformMatroid,
    MutableMatroid,
)
from utils.stopwatch import Stopwatch


rng = np.random.default_rng(seed=2022)


def generate_dummy_matroid(
    size: int, rank: int, uniform_weights: bool = True
) -> MutableMatroid:
    return MutableIntUniformMatroid(size, rank)


def setup(
    size: int, rank: int, *, uniform_weights: bool
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


def time_dynamic(
    algorithm: DynamicMaximalIndependentSetAlgorithm, *args, **kwargs
) -> float:
    """Time one run of the naive dynamic algorithm; return time in seconds."""
    matroid, removal_sequence = setup(*args, **kwargs)

    # start generator (only want to time dynamic part)
    remover = algorithm(matroid)
    remover.send(None)

    with Stopwatch() as stopwatch:
        for element in removal_sequence:
            remover.send(element)

    return stopwatch.measurement


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    x_variable: tp.Literal["size", "rank"]
    x_range: tp.Sequence[int]
    size: int = None
    rank: int = None
    uniform_weights: bool = True

    repeats: int = 5
    normalise_by_size: bool = False

    def __iter__(self) -> tp.Iterable[tp.Dict[str, tp.Any]]:
        """Iterate over the sequence of algorithm inputs."""
        for x in self.x_range:
            yield {
                "size": self.size,
                "rank": self.rank,
                self.x_variable: x,  # overrides one of the above
                "uniform_weights": self.uniform_weights,
            }

    def __len__(self):
        return len(self.x_range)


timers = {
    "restart_greedy": time_restart_greedy,
    "naive_dynamic": functools.partial(
        time_dynamic, dynamic_removal_maximal_independent_set
    ),
    "uniform_weights_dynamic": functools.partial(
        time_dynamic, dynamic_removal_maximal_independent_set_uniform_weights
    ),
}


experiments = {
    "Total time over exhausting sequence of deletions\n"
    f"uniform weights, fixed size ({size}), varying rank": ExperimentConfig(
        x_variable="rank",
        x_range=range(0, size, 20),
        size=size,
    )
    for size in [200]
} | {
    "Time per deletion over exhausting sequence of deletions\n"
    f"uniform weights, varying size, fixed rank ({rank})": ExperimentConfig(
        x_variable="size",
        x_range=range(100, 1000, 100),
        rank=rank,
        normalise_by_size=True,
    )
    for rank in [10, 20, 40]
}


plt.style.use(matplotx.styles.dufte)

for title, config in experiments.items():
    times = np.full(
        shape=(len(timers), len(config.x_range), config.repeats), fill_value=np.nan
    )

    for j, input_data in enumerate(tqdm.tqdm(config)):
        for i, (algorithm_name, timer) in enumerate(timers.items()):
            for k in range(config.repeats):
                times[i, j, k] = timer(**input_data)

    if config.normalise_by_size:
        times /= (
            np.array(config.x_range)[np.newaxis, :, np.newaxis]
            if config.x_variable == "size"
            else config.size
        )

    means = times.mean(axis=-1)
    stds = times.std(axis=-1)

    plt.suptitle(title)
    plt.xlabel(config.x_variable)
    plt.ylabel("time (s)")

    for i, algorithm_name in enumerate(timers):
        plt.errorbar(
            config.x_range, means[i], yerr=stds[i], marker=".", label=algorithm_name
        )

    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.show()
