"""Empirical analysis of the dynamic algorithms on real graph datasets."""

import copy
import functools
import itertools as itt
import random
import typing as tp

from matroids.algorithms.dynamic import (
    DynamicMaximalIndependentSetAlgorithm,
    NaiveDynamic,
    PartialDynamicMaximalIndependentSetAlgorithm,
    RestartGreedy,
    dynamic_removal_maximal_independent_set_uniform_weights,
)
from matroids.matroid import GraphicalMatroid, MutableMatroid, T, set_weights
from utils.performance_experiment import (
    InputData,
    PerformanceExperiment,
    PerformanceExperimentGroup,
)
from utils.seed import set_seed
from utils.slndc import load_facebook_dataset
from utils.stopwatch import Stopwatch


set_seed(2022)


# download a graph dataset from the Stanford Large Network Dataset Collection
# index by number of edges (size of ground set)
fb_dataset = load_facebook_dataset()
networks = {
    num_edges: list(graphs)
    for num_edges, graphs in itt.groupby(fb_dataset, key=lambda g: len(g.edges))
}


def input_generator(
    size: int, uniform_weights: bool, number_of_deletions: int
) -> tp.Iterator[InputData]:
    for graph in itt.cycle(networks[size]):
        graph = copy.deepcopy(graph)
        if not uniform_weights:
            set_weights(graph, {e: random.random() for e in graph.edges})

        matroid = GraphicalMatroid(graph)
        elements = list(matroid.ground_set)

        # reuse same matroid for 5 instances
        for _ in range(5):
            random.shuffle(elements)
            yield {
                "matroid": matroid,
                "removal_sequence": elements[:number_of_deletions],
            }


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


non_uniform_timers = {
    "restart_greedy": functools.partial(time_full_dynamic, RestartGreedy),
    "naive_dynamic": functools.partial(time_full_dynamic, NaiveDynamic),
}


uniform_timers = non_uniform_timers | {
    "uniform_weights_dynamic": functools.partial(
        time_partial_dynamic, dynamic_removal_maximal_independent_set_uniform_weights
    ),
}

PerformanceExperimentGroup(
    identifier="dynamic_fb",
    title=f"Total time over {(num_deletions := 50)} deletions",
    experiments=[
        PerformanceExperiment(
            title=f"uniform_weights = {uniform}",
            timer_functions=uniform_timers if uniform else non_uniform_timers,
            x_name="size",
            x_range=sorted(networks.keys())[:7],  # smallest 7, otherwise takes too long
            fixed_variables={
                "uniform_weights": uniform,
                "number_of_deletions": num_deletions,
            },
            input_generator=input_generator,
            generated_inputs=9,
            repeats=3,
        )
        for uniform in [False, True]
    ],
).measure_show_and_save(
    legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, 0.0)}
)
