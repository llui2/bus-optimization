from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from busline_ga.config.project_paths import PROJECT_ROOT, ROAD_NETWORK_DIR


MapGenerationConfig = Dict[str, float]
GridNode = Tuple[int, int]
VariantInfo = Dict[str, object]


ROAD_VARIANT_VERSION = "orthogonal_grid_v2"
AXIS_GAP_THRESHOLD = 0.50

MAP_GENERATION_CONFIGS: Dict[str, MapGenerationConfig] = {
    "light_fill": {
        "fill_fraction": 0.40,
    },
    "dense_fill": {
        "fill_fraction": 1.00,
    },
}


def canonical_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def get_paths(project_dir: str) -> Tuple[str, str, str, str]:
    road_dir = str(ROAD_NETWORK_DIR)
    nodes_path = os.path.join(road_dir, "nodes.csv")
    edges_path = os.path.join(road_dir, "edges.csv")
    variants_dir = os.path.join(road_dir, "variants")
    metadata_path = os.path.join(variants_dir, "variant_metadata.json")
    os.makedirs(variants_dir, exist_ok=True)
    return nodes_path, edges_path, variants_dir, metadata_path


def load_base_network(
    nodes_path: str,
    edges_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    return nodes_df, edges_df


def cluster_axis_values(values: Iterable[float], gap_threshold: float) -> List[float]:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return []

    groups: List[List[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if value - groups[-1][-1] > gap_threshold:
            groups.append([value])
        else:
            groups[-1].append(value)

    return [sum(group) / len(group) for group in groups]


def nearest_axis_index(value: float, axis_centers: List[float]) -> int:
    return min(range(len(axis_centers)), key=lambda index: abs(value - axis_centers[index]))


def infer_grid_layout(
    nodes_df: pd.DataFrame,
    gap_threshold: float = AXIS_GAP_THRESHOLD,
) -> Tuple[Dict[GridNode, int], Dict[int, GridNode], List[float], List[float]]:
    x_centers = cluster_axis_values(nodes_df["x"], gap_threshold)
    y_centers = cluster_axis_values(nodes_df["y"], gap_threshold)

    grid_to_node: Dict[GridNode, int] = {}
    node_to_grid: Dict[int, GridNode] = {}

    for row in nodes_df.itertuples():
        col_index = nearest_axis_index(float(row.x), x_centers)
        row_index = nearest_axis_index(float(row.y), y_centers)
        grid_key = (col_index, row_index)

        if grid_key in grid_to_node:
            raise ValueError(f"Duplicated grid cell detected for {grid_key}")

        node_id = int(row.node)
        grid_to_node[grid_key] = node_id
        node_to_grid[node_id] = grid_key

    expected_size = len(x_centers) * len(y_centers)
    if len(grid_to_node) != expected_size:
        raise ValueError(
            "The road network does not map cleanly to a full orthogonal grid. "
            f"Assigned={len(grid_to_node)} expected={expected_size}"
        )

    return grid_to_node, node_to_grid, x_centers, y_centers


def build_position_lookup(nodes_df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    return {
        int(row.node): (float(row.x), float(row.y))
        for row in nodes_df.itertuples()
    }


def edge_distance(
    positions: Dict[int, Tuple[float, float]],
    u: int,
    v: int,
) -> float:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    return float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)


def compute_missing_orthogonal_edges(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> List[Tuple[int, int, float]]:
    positions = build_position_lookup(nodes_df)
    grid_to_node, _, x_centers, y_centers = infer_grid_layout(nodes_df)
    existing_edges = {
        canonical_edge(int(row.src), int(row.dst))
        for row in edges_df.itertuples()
    }

    candidates: List[Tuple[int, int, float]] = []
    for col_index in range(len(x_centers)):
        for row_index in range(len(y_centers)):
            u = grid_to_node[(col_index, row_index)]
            for delta_col, delta_row in ((1, 0), (0, 1)):
                neighbor_key = (col_index + delta_col, row_index + delta_row)
                if neighbor_key not in grid_to_node:
                    continue

                v = grid_to_node[neighbor_key]
                edge = canonical_edge(u, v)
                if edge in existing_edges:
                    continue

                distance = edge_distance(positions, u, v)
                candidates.append((u, v, distance))

    candidates.sort(key=lambda item: item[2])
    return candidates


def select_orthogonal_edges(
    candidates: List[Tuple[int, int, float]],
    fill_fraction: float,
) -> List[Tuple[int, int, float]]:
    if not candidates:
        return []

    keep_count = max(1, int(round(len(candidates) * fill_fraction)))
    keep_count = min(keep_count, len(candidates))
    return candidates[:keep_count]


def count_diagonal_synthetic_edges(
    selected_edges: List[Tuple[int, int, float]],
    node_to_grid: Dict[int, GridNode],
) -> int:
    diagonal_count = 0
    for u, v, _ in selected_edges:
        col_u, row_u = node_to_grid[u]
        col_v, row_v = node_to_grid[v]
        if col_u != col_v and row_u != row_v:
            diagonal_count += 1
    return diagonal_count


def add_synthetic_edges(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    fill_fraction: float,
) -> Tuple[pd.DataFrame, int, int]:
    _, node_to_grid, _, _ = infer_grid_layout(nodes_df)
    candidates = compute_missing_orthogonal_edges(nodes_df, edges_df)
    selected_edges = select_orthogonal_edges(candidates, fill_fraction)
    diagonal_count = count_diagonal_synthetic_edges(selected_edges, node_to_grid)

    added_rows = [
        {"src": u, "dst": v, "cost": distance}
        for u, v, distance in selected_edges
    ]
    added_df = pd.DataFrame(added_rows)
    result_edges = pd.concat([edges_df, added_df], ignore_index=True) if added_rows else edges_df.copy()

    return result_edges, len(selected_edges), diagonal_count


def write_variant(
    variant_dir: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> Tuple[str, str]:
    os.makedirs(variant_dir, exist_ok=True)
    nodes_out = os.path.join(variant_dir, "nodes.csv")
    edges_out = os.path.join(variant_dir, "edges.csv")
    nodes_df.to_csv(nodes_out, index=False)
    edges_df.to_csv(edges_out, index=False)
    return nodes_out, edges_out


def write_metadata(metadata_path: str) -> None:
    metadata = {"version": ROAD_VARIANT_VERSION}
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def read_metadata_version(metadata_path: str) -> str | None:
    version = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        version = str(metadata.get("version"))
    return version


def build_variants(project_dir: str, verbose: bool = True) -> Dict[str, VariantInfo]:
    nodes_path, edges_path, variants_dir, metadata_path = get_paths(project_dir)
    nodes_df, base_edges_df = load_base_network(nodes_path, edges_path)
    base_edge_count = len(base_edges_df)
    results: Dict[str, VariantInfo] = {}

    base_dir = os.path.join(variants_dir, "base")
    base_nodes_out, base_edges_out = write_variant(base_dir, nodes_df, base_edges_df)
    results["base"] = {
        "nodes_path": base_nodes_out,
        "edges_path": base_edges_out,
        "nodes": len(nodes_df),
        "edges": base_edge_count,
        "original_edges": base_edge_count,
        "added_edges": 0,
        "diagonal_added_edges": 0,
    }

    for map_case, config in MAP_GENERATION_CONFIGS.items():
        variant_edges_df, added_edges, diagonal_added_edges = add_synthetic_edges(
            nodes_df,
            base_edges_df,
            fill_fraction=float(config["fill_fraction"]),
        )
        variant_dir = os.path.join(variants_dir, map_case)
        nodes_out, edges_out = write_variant(variant_dir, nodes_df, variant_edges_df)
        results[map_case] = {
            "nodes_path": nodes_out,
            "edges_path": edges_out,
            "nodes": len(nodes_df),
            "edges": len(variant_edges_df),
            "original_edges": base_edge_count,
            "added_edges": added_edges,
            "diagonal_added_edges": diagonal_added_edges,
        }

    write_metadata(metadata_path)

    if verbose:
        print("Variants de xarxa generades:")
        for map_case in ["base", "light_fill", "dense_fill"]:
            info = results[map_case]
            print(
                f" - {map_case}: original_edges={info['original_edges']} "
                f"added_edges={info['added_edges']} edges={info['edges']} "
                f"diagonal_added_edges={info['diagonal_added_edges']}"
            )

    return results


def ensure_road_variants(project_dir: str, verbose: bool = True) -> Dict[str, VariantInfo]:
    _, _, variants_dir, metadata_path = get_paths(project_dir)
    required_paths = [
        os.path.join(variants_dir, "base", "nodes.csv"),
        os.path.join(variants_dir, "base", "edges.csv"),
        os.path.join(variants_dir, "light_fill", "nodes.csv"),
        os.path.join(variants_dir, "light_fill", "edges.csv"),
        os.path.join(variants_dir, "dense_fill", "nodes.csv"),
        os.path.join(variants_dir, "dense_fill", "edges.csv"),
    ]

    current_version = read_metadata_version(metadata_path)
    should_generate = (
        not all(os.path.exists(path) for path in required_paths)
        or current_version != ROAD_VARIANT_VERSION
    )

    if should_generate:
        return build_variants(project_dir, verbose=verbose)

    nodes_path, edges_path, _, _ = get_paths(project_dir)
    nodes_df, base_edges_df = load_base_network(nodes_path, edges_path)
    base_edge_count = len(base_edges_df)
    existing_results: Dict[str, VariantInfo] = {}
    for map_case in ["base", "light_fill", "dense_fill"]:
        variant_dir = os.path.join(variants_dir, map_case)
        variant_edges_df = pd.read_csv(os.path.join(variant_dir, "edges.csv"))
        added_edges = max(0, len(variant_edges_df) - base_edge_count)
        existing_results[map_case] = {
            "nodes_path": os.path.join(variant_dir, "nodes.csv"),
            "edges_path": os.path.join(variant_dir, "edges.csv"),
            "nodes": len(nodes_df),
            "edges": len(variant_edges_df),
            "original_edges": base_edge_count,
            "added_edges": added_edges,
            "diagonal_added_edges": 0,
        }

    if verbose:
        print("Variants de xarxa disponibles:")
        for map_case in ["base", "light_fill", "dense_fill"]:
            info = existing_results[map_case]
            print(
                f" - {map_case}: original_edges={info['original_edges']} "
                f"added_edges={info['added_edges']} edges={info['edges']} "
                f"diagonal_added_edges={info['diagonal_added_edges']}"
            )

    return existing_results


def main() -> None:
    build_variants(str(PROJECT_ROOT), verbose=True)


if __name__ == "__main__":
    main()

