from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


Node = int
Edge = Tuple[Node, Node]


@dataclass
class NetworkModel:
    graph: nx.Graph
    bus_stops: List[Node]
    positions: Dict[Node, Tuple[float, float]]
    od_matrix: np.ndarray

    def __post_init__(self) -> None:
        self.node_to_idx: Dict[Node, int] = {
            node: idx for idx, node in enumerate(self.bus_stops)
        }
        self.shortest_paths: Dict[Tuple[Node, Node], List[Node]] = {}
        self.shortest_costs: Dict[Tuple[Node, Node], float] = {}

    def get_stop_index(self, node: Node) -> int:
        return self.node_to_idx[node]

    def get_od_value(self, stop_i: Node, stop_j: Node) -> float:
        i = self.get_stop_index(stop_i)
        j = self.get_stop_index(stop_j)
        return float(self.od_matrix[i, j])

    def get_shortest_path_between_stops(self, stop_i: Node, stop_j: Node) -> List[Node]:
        key = (stop_i, stop_j)

        if key not in self.shortest_paths:
            path = nx.shortest_path(self.graph, source=stop_i, target=stop_j, weight="cost")
            self.shortest_paths[(stop_i, stop_j)] = path
            self.shortest_paths[(stop_j, stop_i)] = list(reversed(path))

        return self.shortest_paths[key]

    def get_shortest_cost_between_stops(self, stop_i: Node, stop_j: Node) -> float:
        key = (stop_i, stop_j)

        if key not in self.shortest_costs:
            cost = nx.shortest_path_length(self.graph, source=stop_i, target=stop_j, weight="cost")
            self.shortest_costs[(stop_i, stop_j)] = float(cost)
            self.shortest_costs[(stop_j, stop_i)] = float(cost)

        return self.shortest_costs[key]

    def get_all_shortest_paths_between_stops(self, stop_i: Node, stop_j: Node) -> List[List[Node]]:
        paths = list(
            nx.all_shortest_paths(
                self.graph,
                source=stop_i,
                target=stop_j,
                weight="cost"
            )
        )
        return paths


def load_network_model(
    road_nodes_path: str,
    road_edges_path: str,
    bus_stops_path: str,
    od_matrix_path: str
) -> NetworkModel:
    nodes_df = pd.read_csv(road_nodes_path)
    edges_df = pd.read_csv(road_edges_path)
    bus_df = pd.read_csv(bus_stops_path)
    od_df = pd.read_csv(od_matrix_path, index_col=0)

    graph = nx.Graph()
    positions: Dict[Node, Tuple[float, float]] = {}

    for _, row in nodes_df.iterrows():
        node = int(row["node"])
        x = float(row["x"])
        y = float(row["y"])
        positions[node] = (x, y)
        graph.add_node(node, pos=(x, y))

    for _, row in edges_df.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])

        if "cost" in edges_df.columns:
            cost = float(row["cost"])
        else:
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            cost = float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

        graph.add_edge(src, dst, cost=cost)

    bus_stops = [int(node) for node in bus_df["node"].tolist()]
    od_matrix = od_df.to_numpy(dtype=float)

    model = NetworkModel(
        graph=graph,
        bus_stops=bus_stops,
        positions=positions,
        od_matrix=od_matrix
    )

    return model
