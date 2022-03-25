import itertools
import typing as tp

import networkx as nx

from matroids.matroid import EdgeType


def compute_missing_edges(
    graph: nx.Graph, extra_nodes: tp.Set[tp.Any] = None
) -> tp.Set[EdgeType]:
    """
    Compute the set of edges that can be added to the graph.

    :param graph: Graph object.
    :param extra_nodes: Extra nodes outside of the graph from which to consider edges.
        Default is none.

    :return: A set of missing edges.
    """
    extra_nodes = extra_nodes if extra_nodes is not None else set()
    all_edges = set(itertools.combinations(graph.nodes | extra_nodes, r=2))  # O(|V|^2)
    missing_edges = all_edges - set(graph.edges)
    return missing_edges
