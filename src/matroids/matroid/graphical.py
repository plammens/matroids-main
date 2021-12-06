import dataclasses
import typing

import networkx as nx

from .base import MutableMatroid

EdgeType = typing.Tuple[typing.Any, typing.Any]


@dataclasses.dataclass(frozen=True)
class GraphicalMatroid(MutableMatroid[EdgeType]):
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
    def ground_set(self) -> typing.Collection[EdgeType]:
        return self.graph.edges

    def __bool__(self):
        return bool(self.graph.edges)

    def is_independent(self, subset: typing.AbstractSet[EdgeType]) -> bool:
        subgraph = self.graph.edge_subgraph(subset)
        if not subgraph:
            return True  # special case for empty graph; otherwise nx exception
        return nx.algorithms.tree.is_forest(subgraph)

    class StatefulIndependenceChecker(MutableMatroid.StatefulIndependenceChecker):
        def __init__(
            self,
            matroid: "GraphicalMatroid",
            independent_subset: typing.MutableSet[EdgeType],
        ):
            super().__init__(matroid, independent_subset)
            self.matroid: "GraphicalMatroid"

            # initialise a disjoint-set data structure
            # for determining the connected component that a given node belongs to
            initial_subgraph = self.matroid.graph.edge_subgraph(independent_subset)
            node_to_connected_component = nx.utils.UnionFind(initial_subgraph.nodes)
            connected_components = nx.algorithms.connected_components(initial_subgraph)
            for component_nodes in connected_components:
                node_to_connected_component.union(*component_nodes)

            self.node_to_connected_component = node_to_connected_component

        def would_be_independent_after_adding(self, element: EdgeType) -> bool:
            u, v = element
            # the subset remains independent iff adding the edge {u, v} doesn't add a
            # cycle, i.e. if {u, v} connects two different connected components.
            # but if the edge was already in the set, adding it won't change anything
            same_components = (
                self.node_to_connected_component[u]
                == self.node_to_connected_component[v]
            )
            return not same_components or element in self.independent_subset

        def add_element(self, element: EdgeType) -> None:
            u, v = element
            super().add_element(element)
            # update the connected components info (merge the two components)
            self.node_to_connected_component.union(u, v)

    def get_weight(self, element: EdgeType) -> float:
        weight = self.graph.get_edge_data(*element).get("weight", 1.0)
        assert isinstance(weight, float)
        return weight

    def add_element(self, element: EdgeType) -> None:
        self.graph.add_edge(*element)

    def remove_element(self, element: EdgeType) -> None:
        self.graph.remove_edge(*element)


def set_weights(graph: nx.Graph, weights: typing.Mapping[EdgeType, float]) -> None:
    """Utility to set weights on a graph in a way compatible with GraphicalMatroid."""
    for (u, v), w in weights.items():
        graph[u][v]["weight"] = w
