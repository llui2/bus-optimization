from __future__ import annotations

from typing import Dict, List, Tuple

from busline_ga.core.network_model import Edge, NetworkModel, Node


def canonical_edge(u: Node, v: Node) -> Edge:
    return (u, v) if u <= v else (v, u)


def path_to_edges(path: List[Node]) -> List[Edge]:
    edges: List[Edge] = []
    for i in range(len(path) - 1):
        edges.append(canonical_edge(path[i], path[i + 1]))
    return edges

def remove_backtracking(nodes):
    result = []

    for node in nodes:
        if len(result) >= 2 and node == result[-2]:
            # detecta patrÃ³ A â†’ B â†’ A i elimina B
            result.pop()
        else:
            result.append(node)

    return result

def build_line_from_stops(network: NetworkModel, stops: List[Node]) -> Dict[str, List]:
    """
    Construeix la lÃ­nia real sobre la xarxa viÃ ria a partir d'una seqÃ¼Ã¨ncia de parades.
    """
    line_paths: List[List[Node]] = []
    line_edges: List[Edge] = []
    line_nodes_real: List[Node] = []

    if len(stops) < 2:
        result = {
            "stops": stops,
            "paths": line_paths,
            "line_nodes_real": line_nodes_real,
            "line_edges": line_edges,
        }
    else:
        for i in range(len(stops) - 1):
            path = network.get_shortest_path_between_stops(stops[i], stops[i + 1])
            line_paths.append(path)

            if not line_nodes_real:
                line_nodes_real.extend(path)
            else:
                # Evitem duplicar el node inicial del segÃ¼ent subcamÃ­
                line_nodes_real.extend(path[1:])

        line_edges = path_to_edges(line_nodes_real)

        result = {
            "stops": stops,
            "paths": line_paths,
            "line_nodes_real": line_nodes_real,
            "line_edges": line_edges,
        }

    return result

