import dataclasses
import typing

from .base import Matroid

import networkx as nx

EdgeType = typing.Tuple[typing.Any, typing.Any]


@dataclasses.dataclass(frozen=True)
class GraphicalMatroid(Matroid[EdgeType]):
    """
    A matroid based on a graph.

    If G = (V, E) is a graph, its corresponding graphical matroid is (E, I) where I
    is the set of subsets of edges (subsets of E) such that (V, E) is a forest (i.e.
    doesn't contain any cycles).

    The weights are inferred from a(n expected) ``"weight"`` attribute of the edges
    in the given NetworkX graph object.
    """

    graph: nx.Graph

    @property
    def ground_set(self) -> typing.FrozenSet[EdgeType]:
        return frozenset(self.graph.edges)

    def is_independent(self, subset: typing.Collection[EdgeType]) -> bool:
        subgraph = self.graph.edge_subgraph(subset)
        if not subgraph:
            return True  # special case for empty graph; otherwise nx exception
        return nx.algorithms.tree.is_forest(subgraph)

    def get_weight(self, element: EdgeType) -> float:
        weight = self.graph.get_edge_data(*element).get("weight", 1.0)
        assert isinstance(weight, float)
        return weight
