#!/usr/bin/env python3
"""Empirical analysis of the greedy algorithm on real graph datasets."""

import itertools as itt
import typing as tp

from matroids.algorithms.static import maximal_independent_set
from matroids.matroid import GraphicalMatroid
from utils.performance_experiment import (
    InputData,
    PerformanceExperiment,
    PerformanceExperimentGroup,
)
from utils.seed import set_seed
from utils.slndc import load_facebook_dataset
from utils.stopwatch import make_timer


set_seed(2022)


# download a graph dataset from the Stanford Large Network Dataset Collection
# index by number of edges (size of ground set)
fb_dataset = load_facebook_dataset()
networks = {
    num_edges: list(graphs)
    for num_edges, graphs in itt.groupby(fb_dataset, key=lambda g: len(g.edges))
}


def input_generator(size: int) -> tp.Iterator[InputData]:
    for graph in itt.cycle(networks[size]):
        yield {"matroid": GraphicalMatroid(graph)}


timers = {"greedy": make_timer(maximal_independent_set)}


PerformanceExperimentGroup(
    identifier="greedy_fb",
    title="Performance on the Facebook dataset",
    experiments=[
        PerformanceExperiment(
            timer_functions=timers,
            x_name="size",
            x_range=sorted(networks.keys()),
            input_generator=input_generator,
            generated_inputs=5,
        )
    ],
).measure_show_and_save(legend_kwargs={"loc": "upper center"})
