#!/usr/bin/env python3
"""Script to generate some performance graphs for the greedy algorithm."""

import functools
import itertools as itt
import more_itertools as mitt

import numpy as np

from matroids.algorithms.static import maximal_independent_set
from matroids.matroid.uniform import IntUniformMatroid
from utils.generate import (
    generate_2_by_n_matrix_matroid,
    generate_n_by_n_matrix_matroid,
    generate_random_dummy_matroid,
)
from utils.performance_experiment import (
    PerformanceExperiment,
    PerformanceExperimentGroup,
)
from utils.seed import set_seed
from utils.stopwatch import make_timer


set_seed(2022)


timers = {"greedy": make_timer(maximal_independent_set)}


PerformanceExperimentGroup(
    identifier="greedy_uniform_matroid",
    title="Performance on uniform matroid",
    experiments=[
        PerformanceExperiment(
            title=f"rank = {rank}",
            timer_functions=timers,
            x_name="size",
            x_range=np.linspace(0, 10_000, num=10, dtype=int),
            input_generator=lambda size: (
                {"matroid": m}
                for m in mitt.repeatfunc(
                    functools.partial(
                        generate_random_dummy_matroid, size, rank, uniform_weights=False
                    )
                )
            ),
            generated_inputs=5,
        )
        for rank in [50, 1000]
    ],
).measure_show_and_save(
    legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, 0.0)}
)


PerformanceExperimentGroup(
    identifier="greedy_free_matroid",
    title="Performance on free matroid",
    experiments=[
        PerformanceExperiment(
            timer_functions=timers,
            x_name="size",
            x_range=np.linspace(0, 100, num=10, dtype=int),
            input_generator=lambda size: itt.repeat(
                {"matroid": IntUniformMatroid.free(size)}
            ),
            generated_inputs=1,
        )
    ],
).measure_show_and_save(legend_kwargs={"loc": "upper center"})


PerformanceExperimentGroup(
    identifier="greedy_2_by_n",
    title="Performance on 2 x n random matrices",
    experiments=[
        PerformanceExperiment(
            timer_functions=timers,
            x_name="n",
            x_range=np.linspace(0, 1000, num=10, dtype=int),
            input_generator=lambda n: (
                {"matroid": m}
                for m in mitt.repeatfunc(
                    functools.partial(generate_2_by_n_matrix_matroid, n)
                )
            ),
            generated_inputs=10,
        )
    ],
).measure_show_and_save(legend_kwargs={"loc": "upper center"})


PerformanceExperimentGroup(
    identifier="greedy_n_by_n",
    title="Performance on n x n random matrices",
    experiments=[
        PerformanceExperiment(
            timer_functions=timers,
            x_name="n",
            x_range=np.linspace(0, 150, num=15, dtype=int),
            input_generator=lambda n: (
                {"matroid": m}
                for m in mitt.repeatfunc(
                    functools.partial(generate_n_by_n_matrix_matroid, n)
                )
            ),
            generated_inputs=10,
        )
    ],
).measure_show_and_save(legend_kwargs={"loc": "upper center"})
