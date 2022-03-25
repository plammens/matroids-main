import itertools
import os
import pathlib
import typing as tp

import networkx as nx

from matroids.matroid import EdgeType


ROOT_OUTPUT_PATH = pathlib.Path(__file__).parent.parent.parent.resolve() / "artifacts"


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


def ensure_output_dir_exists(path: os.PathLike = ROOT_OUTPUT_PATH) -> pathlib.Path:
    """Create any missing directories in the given path."""
    path = pathlib.Path(path)
    if not path.exists():
        ensure_output_dir_exists(path.parent)
        os.mkdir(path)
    elif not path.is_dir():
        raise IOError("Output path {} already exists and is a file".format(path))
    return path
