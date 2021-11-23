"""Empirical analysis of the greedy algorithm on real graph datasets."""

import matplotlib.pyplot as plt
import numpy as np
import perfplot

from matroids.algorithms.greedy import maximal_independent_set
from matroids.matroid import GraphicalMatroid
from utils.slndc import load_facebook_dataset


rng = np.random.default_rng(seed=2021)

# download a graph dataset from the Stanford Large Network Dataset Collection
# index by number of edges (size of ground set)
networks = {len(g.edges): g for g in load_facebook_dataset()}

plots = {
    "Performance on the Facebook dataset": (
        sorted(networks.keys())[:7],
        lambda n: GraphicalMatroid(networks[n]),
    ),
}

for title, (n_range, factory) in plots.items():
    results = perfplot.bench(
        n_range=list(n_range),
        setup=factory,
        kernels=[maximal_independent_set],
        labels=["greedy"],
        xlabel="number of edges",
    )
    results.plot()
    plt.title(title)
    plt.tight_layout()
    plt.show()
