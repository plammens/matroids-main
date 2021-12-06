"""Script to generate some performance graphs for the greedy algorithm."""
import matplotlib.pyplot as plt
import perfplot
import numpy as np

from matroids.matroid import RealLinearMatroid, ExplicitMatroid
from matroids.algorithms.static import maximal_independent_set


rng = np.random.default_rng(seed=2021)


def generate_2_by_n_matroid(size: int) -> RealLinearMatroid:
    matrix = rng.random((2, size))
    weights = rng.random((size,))
    return RealLinearMatroid(matrix, weights)


def generate_n_by_n_matroid(size: int) -> RealLinearMatroid:
    matrix = rng.random((size, size))
    weights = rng.random((size,))
    return RealLinearMatroid(matrix, weights)


plots = {
    "Performance on $U_{3,n}$": (
        range(0, 100, 10),
        lambda n: ExplicitMatroid.uniform(elements=range(n), k=3),
    ),
    "Performance on free matroid of size n": (
        range(0, 20, 1),
        lambda n: ExplicitMatroid.uniform(elements=range(n), k=n),
    ),
    "Performance on 2 x n uniform random matrices": (
        range(0, 1000, 100),
        generate_2_by_n_matroid,
    ),
    "Performance on n x n uniform random matrices": (
        range(0, 100, 10),
        generate_n_by_n_matroid,
    ),
}

for title, (n_range, factory) in plots.items():
    results = perfplot.bench(
        n_range=list(n_range),
        setup=factory,
        kernels=[maximal_independent_set],
        labels=["greedy"],
        xlabel="n = |E|",
    )
    results.plot()
    plt.title(title)
    plt.tight_layout()
    plt.show()
