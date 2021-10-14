import dataclasses
import itertools
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

    def is_independent(self, subset: typing.AbstractSet[EdgeType]) -> bool:
        subgraph = self.graph.edge_subgraph(subset)
        if not subgraph:
            return True  # special case for empty graph; otherwise nx exception
        return nx.algorithms.tree.is_forest(subgraph)

    def is_independent_incremental_stateful(
        self, independent_subset: typing.MutableSet[EdgeType]
    ) -> typing.Generator[bool, EdgeType, None]:
        # initialise a disjoint-set data structure
        # for determining the connected component that a given node belongs to
        initial_subgraph = self.graph.edge_subgraph(independent_subset)
        node_to_connected_component = nx.utils.UnionFind(initial_subgraph.nodes)
        connected_components = nx.algorithms.connected_components(initial_subgraph)
        for c in connected_components:
            node_to_connected_component.union(*c.nodes)

        # the generator
        is_independent = True
        while True:
            u, v = new_edge = (yield is_independent)
            # the following check is amortized O(1)
            if node_to_connected_component[u] != node_to_connected_component[v]:
                # the edge connects different connected components, so we can add it
                is_independent = True
                independent_subset.add(new_edge)

    def get_weight(self, element: EdgeType) -> float:
        weight = self.graph.get_edge_data(*element).get("weight", 1.0)
        assert isinstance(weight, float)
        return weight
