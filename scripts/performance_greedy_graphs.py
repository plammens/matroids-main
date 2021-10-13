"""Empirical analysis of the greedy algorithm on real graph datasets."""
import pathlib
import tarfile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import perfplot

from matroids.algorithms.greedy import maximal_independent_set
from matroids.matroid import GraphicalMatroid
from utils.download import ensure_downloaded


rng = np.random.default_rng(seed=2021)


# download a graph dataset from the Stanford Large Network Dataset Collection
path = ensure_downloaded(
    "https://snap.stanford.edu/data/facebook.tar.gz",
    path=pathlib.Path.cwd().joinpath("downloads").joinpath("facebook.tar.gz"),
)
tar = tarfile.open(path)
filenames = [name for name in tar.getnames() if name.endswith("edges")]
networks = []
for name in filenames:
    with tar.extractfile(name) as file:
        edges = [tuple(map(int, line.split())) for line in file.readlines()]
        network = nx.from_edgelist(edges)
        networks.append(network)
networks = dict((len(g.edges), g) for g in networks)


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
