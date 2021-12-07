"""Empirical analysis of the naive dynamic algorithm on real graph datasets."""
import operator
from typing import Callable, Generator, List, Mapping, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import perfplot

from matroids.algorithms.dynamic import dynamic_removal_maximal_independent_set
from matroids.algorithms.static import maximal_independent_set
from matroids.matroid import EdgeType, GraphicalMatroid, MutableMatroid, set_weights
from utils.slndc import load_facebook_dataset


rng = np.random.default_rng(seed=2021)

# download a graph dataset from the Stanford Large Network Dataset Collection
# index by number of edges (size of ground set)
networks = {len(g.edges): g for g in load_facebook_dataset()}


# define the setup function, mapping input size to setup data

SetupData = Tuple[MutableMatroid, MutableMatroid, List[EdgeType], Generator]


def make_network_copy(network: nx.Graph, weights: Mapping[EdgeType, float]) -> nx.Graph:
    copy = network.copy()
    set_weights(copy, weights)
    return copy


def make_setup(
    uniform_weights: bool, number_of_deletions: int
) -> Callable[[int], SetupData]:
    def setup(n: int) -> SetupData:
        network: nx.Graph = networks[n]
        weights = {} if uniform_weights else {e: rng.random() for e in network.edges}
        matroid_copy1 = GraphicalMatroid(make_network_copy(network, weights))
        matroid_copy2 = GraphicalMatroid(make_network_copy(network, weights))

        # start generator for naive dynamic algorithm so as to only test removal step
        remover_generator = dynamic_removal_maximal_independent_set(matroid_copy2)
        maximal_set = remover_generator.send(None)  # compute current MIS

        # choose an element to remove from the maximal independent set
        # (much more interesting than removing another element, which is trivial)
        selected_elements = list(maximal_set)
        # not using rng.choice here because it returns a 2D array (tuples get converted)
        rng.shuffle(selected_elements)
        elements_to_remove = selected_elements[:number_of_deletions]

        return matroid_copy1, matroid_copy2, elements_to_remove, remover_generator

    return setup


# define the two kernels


def restart_greedy(setup_data: SetupData):
    matroid, _, elements_to_remove, *_ = setup_data
    results = []
    for element in elements_to_remove:
        matroid.remove_element(element)
        results.append(maximal_independent_set(matroid))
    return results


def naive_dynamic(setup_data: SetupData):
    _, _, elements_to_remove, remover_generator = setup_data
    return [remover_generator.send(element) for element in elements_to_remove]


plots = {
    "Single deletion on the FB dataset, uniform weights": (
        sorted(networks.keys())[:7],
        make_setup(uniform_weights=True, number_of_deletions=1),
    ),
    "Single deletion on the FB dataset, random weights": (
        sorted(networks.keys())[:7],
        make_setup(uniform_weights=False, number_of_deletions=1),
    ),
    "10 deletions in sequence on the FB dataset, random weights": (
        sorted(networks.keys())[:7],
        make_setup(uniform_weights=False, number_of_deletions=10),
    ),
}

for title, (n_range, factory) in plots.items():
    results = perfplot.bench(
        n_range=list(n_range),
        setup=factory,
        kernels=[restart_greedy, naive_dynamic],
        xlabel="number of edges",
        target_time_per_measurement=0.0,  # avoid repetitions, mutable operations
        equality_check=operator.eq,  # plain equality, can't use np.allclose on sets
    )
    results.plot()
    plt.title(title)
    plt.tight_layout()
    plt.show()
