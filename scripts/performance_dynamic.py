import itertools
from typing import Generator, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import perfplot

from matroids.algorithms.dynamic import (
    dynamic_removal_maximal_independent_set,
    dynamic_removal_maximal_independent_set_uniform_weights,
)
from matroids.algorithms.static import maximal_independent_set
from matroids.matroid import EdgeType, GraphicalMatroid, MutableMatroid


rng = np.random.default_rng(seed=2022)


def generate_graphical_matroid(
    size: int, rank: int, uniform_weights: bool = True
) -> GraphicalMatroid:
    """Generate an arbitrary graphical matroid of the given size and rank."""
    rng = np.random.default_rng(seed=2022)  # local rng for deterministic results

    # start with an underlying tree/forest with a number of edges equal to the rank
    graph: nx.Graph = nx.path_graph(n=rank + 1)
    assert len(graph.edges) == rank

    # add edges until we reach given size
    # set of all possible edges to add:
    possible_edges = list(set(itertools.combinations(graph.nodes, r=2)) - graph.edges)
    rng.shuffle(possible_edges)
    to_add = size - len(graph.edges)
    if to_add > len(possible_edges):
        raise ValueError(
            f"This method is unable to generate a graphical matroid"
            f" of size {size} and rank {rank}"
        )
    graph.add_edges_from(possible_edges[: size - len(graph.edges)])

    matroid = GraphicalMatroid(graph)
    assert len(matroid.ground_set) == size
    assert len(maximal_independent_set(matroid)) == rank
    return matroid


SetupData = Tuple[
    List[EdgeType],
    MutableMatroid,
    MutableMatroid,
    Generator,
    MutableMatroid,
    Generator,
]


def setup(size: int, rank: int, *, uniform_weights: bool) -> SetupData:
    matroid_copy1 = generate_graphical_matroid(size, rank, uniform_weights)
    matroid_copy2 = GraphicalMatroid(matroid_copy1.graph.copy())
    matroid_copy3 = GraphicalMatroid(matroid_copy1.graph.copy())

    # start generators for the dynamic algorithms so as to only benchmark
    # the dynamic removal steps
    remover2 = dynamic_removal_maximal_independent_set(matroid_copy2)
    remover2.send(None)
    remover3 = dynamic_removal_maximal_independent_set_uniform_weights(matroid_copy3)
    remover3.send(None)

    # select the sequence of removals (over all elements of the matroid)
    elements_to_remove = list(matroid_copy1.ground_set)
    rng.shuffle(elements_to_remove)

    return (
        elements_to_remove,
        matroid_copy1,
        matroid_copy2,
        remover2,
        matroid_copy3,
        remover3,
    )


def restart_greedy(setup_data: SetupData):
    elements_to_remove, matroid, *_ = setup_data
    results = []
    for element in elements_to_remove:
        matroid.remove_element(element)
        results.append(maximal_independent_set(matroid))
    return results


def naive_dynamic(setup_data: SetupData):
    elements_to_remove, _, _, remover_generator, *_ = setup_data
    return [remover_generator.send(element) for element in elements_to_remove]


def uniform_weights_dynamic(setup_data: SetupData):
    elements_to_remove, *_, remover_generator = setup_data
    return [remover_generator.send(element) for element in elements_to_remove]

plots = {
    "Total time over exhausting sequence of deletions\n"
    "uniform weights, fixed size (200), varying rank": (
        "k (rank of matroid)",
        range(20, 200, 20),
        lambda k: setup(size=200, rank=k, uniform_weights=True),
        [restart_greedy, naive_dynamic, uniform_weights_dynamic],
    ),
    "Total time over exhausting sequence of deletions\n"
    "uniform weights, varying size, fixed rank (50)": (
        "n (size of matroid)",
        range(100, 1000, 100),
        lambda n: setup(size=n, rank=50, uniform_weights=True),
        [restart_greedy, naive_dynamic, uniform_weights_dynamic],
    ),
}

for title, (x_label, n_range, factory, kernels) in plots.items():
    results = perfplot.bench(
        n_range=list(n_range),
        setup=factory,
        kernels=kernels,
        xlabel=x_label,
        target_time_per_measurement=0.0,  # avoid repetitions, mutable operations
        equality_check=None,  # for speed
    )
    results.plot(logx=False, logy=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
