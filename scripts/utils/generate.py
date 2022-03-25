import random

import networkx as nx

from matroids.matroid import (
    GraphicalMatroid,
    MutableIntUniformMatroid,
    set_weights,
)
from utils.misc import compute_missing_edges


def generate_random_dummy_matroid(
    size: int, rank: int, *, uniform_weights: bool
) -> MutableIntUniformMatroid:
    weights = {} if uniform_weights else {i: random.random() for i in range(size)}
    return MutableIntUniformMatroid(size, rank, weights)


def generate_random_graphical_matroid(
    size: int, rank: int, *, uniform_weights: bool
) -> GraphicalMatroid:
    # we want the maximal spanning tree to have ``rank`` edges
    # a tree of n nodes has (n - 1) edges
    num_vertices = rank + 1
    if size > (num_vertices * (num_vertices - 1) / 2):
        raise ValueError(f"Can't generate graph of size {size} with rank {rank}")
    graph: nx.Graph = nx.random_tree(n=num_vertices)

    # add random cycles until we reach the desired size
    num_missing_edges = size - len(graph.edges)
    missing_edges = compute_missing_edges(graph)
    edges_to_add = random.sample(missing_edges, k=num_missing_edges)
    graph.add_edges_from(edges_to_add)

    # add weights
    if not uniform_weights:
        weights = {edge: random.random() for edge in graph.edges}
        set_weights(graph, weights)

    return GraphicalMatroid(graph)
