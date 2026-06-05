from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from busline_ga.config.project_paths import BUS_NETWORK_DIR, OD_SCENARIOS_DIR, ROAD_NETWORK_DIR


RNG_SEED = 42
TOP_FRACTION = 0.12
CLUSTER_A_SHARE = 0.64
CLUSTER_B_SHARE = 0.30
BRIDGE_SHARE = 0.06
BRIDGE_PAIR_COUNT = 1
CLUSTER_A_TOP_FRACTION = 0.12
CLUSTER_B_TOP_FRACTION = 0.08
SIGMA_A_FACTOR = 0.20
SIGMA_B_FACTOR = 0.16
CENTER_STOP_COUNT = 3
PERIPHERY_FRACTION = 0.45
SECTOR_COUNT = 3
SECTOR_SHARES = [0.50, 0.30, 0.20]
MIN_ACTIVE_PERIPHERAL_STOPS_PER_SECTOR = 2
MAX_ACTIVE_PERIPHERAL_STOPS_PER_SECTOR = 5
ACTIVE_PERIPHERAL_FRACTION = 0.35
CENTER_LINKS_PER_PERIPHERAL_STOP = 1
RADIAL_DEMAND_SHARE = 0.86
CORE_INTERNAL_SHARE = 0.10
LOCAL_PERIPHERAL_SHARE = 0.04
CENTRAL_CANDIDATE_FRACTION = 0.35
C_SECTOR_COUNT = 8
C_GAP_SECTOR_COUNT = 2
ACTIVE_CORE_STOP_COUNT = 10
CORE_NEIGHBORS_PER_STOP = 2
HOTSPOT_STOP_COUNT = 1
HOTSPOT_LINK_COUNT = 3
CORE_SHARE = 0.82
HOTSPOT_SHARE = 0.18
ONE_CENTER_CENTER_STOP_COUNT = 4
EXTERNAL_BRANCH_COUNT = 2
MIN_BRANCH_STOP_COUNT = 2
MAX_BRANCH_STOP_COUNT = 4
ONE_CENTER_CORE_SHARE = 0.55
BRANCH_1_SHARE = 0.25
BRANCH_2_SHARE = 0.15
CROSS_SHARE = 0.05
DENSE_HOTSPOTS_BACKGROUND_DEMAND = 0.1
DENSE_HOTSPOTS_SEED = 2026
DENSE_HOTSPOTS_NEIGHBOR_PAIR_LIMIT = 24
DENSE_HOTSPOTS_NEIGHBORS_PER_ENDPOINT = 2
BALANCED_BACKGROUND_DEMAND = 0.1


def get_paths() -> Tuple[str, str, str]:
    base_od_path = os.path.join(str(BUS_NETWORK_DIR), "od_matrix_fixed.csv")
    road_nodes_path = os.path.join(str(ROAD_NETWORK_DIR), "nodes.csv")
    scenario_dir = str(OD_SCENARIOS_DIR)
    os.makedirs(scenario_dir, exist_ok=True)
    return base_od_path, road_nodes_path, scenario_dir


def load_reference_data(
    base_od_path: str,
    road_nodes_path: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    base_df = pd.read_csv(base_od_path, index_col=0)
    base_df.index = base_df.index.astype(int)
    base_df.columns = base_df.columns.astype(int)

    road_nodes_df = pd.read_csv(road_nodes_path).set_index("node")
    ordered_stops = [int(stop) for stop in base_df.index.tolist()]
    coordinates = road_nodes_df.loc[ordered_stops, ["x", "y"]].to_numpy(dtype=float)
    return base_df, coordinates


def zero_diagonal(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype=float, copy=True)
    np.fill_diagonal(result, 0.0)
    return result


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    result = 0.5 * (matrix + matrix.T)
    return zero_diagonal(result)


def scale_to_total(matrix: np.ndarray, target_total: float) -> np.ndarray:
    result = zero_diagonal(np.maximum(matrix, 0.0))
    current_total = float(result.sum())

    if current_total > 0.0:
        result = result * (target_total / current_total)

    return zero_diagonal(result)


def matrix_to_dataframe(matrix: np.ndarray, template_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=template_df.index, columns=template_df.columns)


def off_diagonal_mass(matrix: np.ndarray) -> float:
    return float(matrix.sum() - np.trace(matrix))


def nonzero_upper_pairs(matrix: np.ndarray) -> int:
    upper = np.triu(matrix, k=1)
    return int(np.count_nonzero(upper > 0.0))


def matrix_density(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    possible_pairs = n * (n - 1) / 2.0
    density = 0.0

    if possible_pairs > 0.0:
        density = nonzero_upper_pairs(matrix) / possible_pairs

    return float(density)


def validate_matrix(matrix: np.ndarray) -> Dict[str, object]:
    return {
        "shape": matrix.shape,
        "symmetric": bool(np.allclose(matrix, matrix.T)),
        "zero_diagonal": bool(np.allclose(np.diag(matrix), 0.0)),
        "min": float(matrix.min()),
        "max": float(matrix.max()),
    }


def print_validation(
    name: str,
    matrix: np.ndarray,
    total_demand: float,
    mass_before_sparsification: float,
) -> None:
    validation = validate_matrix(matrix)
    mass_after = off_diagonal_mass(matrix)
    print(
        f"{name}: shape={validation['shape']} "
        f"symmetric={validation['symmetric']} "
        f"zero_diagonal={validation['zero_diagonal']} "
        f"min={validation['min']:.6f} max={validation['max']:.6f} "
        f"total_demand={total_demand:.6f} "
        f"mass_before={mass_before_sparsification:.6f} "
        f"mass_after={mass_after:.6f} "
        f"nonzero_pairs={nonzero_upper_pairs(matrix)} "
        f"density={matrix_density(matrix):.6f}"
    )


def sparsify_top_pairs(
    matrix: np.ndarray,
    top_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    upper_indices = np.triu_indices_from(matrix, k=1)
    upper_values = matrix[upper_indices]
    n_pairs = len(upper_values)
    keep_count = max(1, int(np.ceil(top_fraction * n_pairs)))

    tie_break = rng.uniform(0.0, 1e-9, size=n_pairs)
    ranking = np.argsort(-(upper_values + tie_break))
    keep_indices = ranking[:keep_count]

    sparse_upper = np.zeros_like(upper_values)
    sparse_upper[keep_indices] = upper_values[keep_indices]

    sparse_matrix = np.zeros_like(matrix)
    sparse_matrix[upper_indices] = sparse_upper
    sparse_matrix = sparse_matrix + sparse_matrix.T
    sparse_matrix = zero_diagonal(sparse_matrix)
    return sparse_matrix


def keep_top_upper_pairs_by_mask(
    matrix: np.ndarray,
    upper_mask: np.ndarray,
    keep_count: int,
) -> np.ndarray:
    result = np.zeros_like(matrix, dtype=float)
    selected_pairs: List[Tuple[int, int, float]] = []

    for row in range(matrix.shape[0]):
        for col in range(row + 1, matrix.shape[1]):
            if upper_mask[row, col]:
                selected_pairs.append((row, col, float(matrix[row, col])))

    selected_pairs.sort(key=lambda item: item[2], reverse=True)
    for row, col, value in selected_pairs[: max(0, keep_count)]:
        result[row, col] = value
        result[col, row] = value

    return zero_diagonal(result)


def sparsify_top_pairs_by_mask(
    matrix: np.ndarray,
    upper_mask: np.ndarray,
    top_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.zeros_like(matrix, dtype=float)
    selected_pairs: List[Tuple[int, int, float]] = []

    for row in range(matrix.shape[0]):
        for col in range(row + 1, matrix.shape[1]):
            if upper_mask[row, col]:
                selected_pairs.append((row, col, float(matrix[row, col])))

    if selected_pairs:
        keep_count = max(1, int(np.ceil(top_fraction * len(selected_pairs))))
        tie_break = rng.uniform(0.0, 1e-9, size=len(selected_pairs))
        ranked_pairs = [
            (row, col, value, value + tie_break[index])
            for index, (row, col, value) in enumerate(selected_pairs)
        ]
        ranked_pairs.sort(key=lambda item: item[3], reverse=True)

        for row, col, value, _ in ranked_pairs[:keep_count]:
            result[row, col] = value
            result[col, row] = value

    return zero_diagonal(result)


def geometric_center_index(coordinates: np.ndarray) -> int:
    center = coordinates.mean(axis=0)
    distances = np.linalg.norm(coordinates - center, axis=1)
    return int(np.argmin(distances))


def select_two_centers(coordinates: np.ndarray) -> Tuple[int, int]:
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    center_i, center_j = np.unravel_index(np.argmax(distances), distances.shape)
    return int(center_i), int(center_j)


def gaussian_affinity(
    coordinates: np.ndarray,
    center_index: int,
    sigma_scale: float = 0.22,
) -> np.ndarray:
    distances = np.linalg.norm(coordinates - coordinates[center_index], axis=1)
    max_distance = float(
        np.max(np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2))
    )
    sigma = max(max_distance * sigma_scale, 1e-6)
    affinity = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
    return affinity


def principal_axis(coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, np.argmax(eigenvalues)]
    orth_axis = np.array([-axis[1], axis[0]])
    return axis, orth_axis


def angle_difference(angle_a: float, angle_b: float) -> float:
    result = abs(math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b)))
    return result


def _pair_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    result = np.sqrt((difference ** 2).sum(axis=2))
    return result


def _top_base_pairs(base_matrix: np.ndarray, limit: int) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[int, int, float]] = []

    for row in range(base_matrix.shape[0]):
        for col in range(row + 1, base_matrix.shape[1]):
            value = float(base_matrix[row, col])
            if value > 0.0:
                pairs.append((row, col, value))

    pairs.sort(key=lambda item: item[2], reverse=True)
    result = pairs[:limit]
    return result


def _amplify_dense_hotspot_base_value(base_value: float) -> float:
    if base_value <= 0.0:
        amplified_value = DENSE_HOTSPOTS_BACKGROUND_DEMAND
    elif base_value <= 0.4:
        amplified_value = 2.0 + ((base_value - 0.1) / 0.3) * 3.0
    else:
        amplified_value = 7.5 * base_value

    result = min(70.0, max(DENSE_HOTSPOTS_BACKGROUND_DEMAND, amplified_value))
    return result


def _nearest_stop_indices(
    stop_index: int,
    distances: np.ndarray,
    excluded_indices: set,
    limit: int,
) -> List[int]:
    ranked_indices = np.argsort(distances[stop_index])
    nearest_indices: List[int] = []

    for candidate_index in ranked_indices:
        candidate = int(candidate_index)
        if candidate == stop_index or candidate in excluded_indices:
            continue

        nearest_indices.append(candidate)
        if len(nearest_indices) >= limit:
            break

    result = nearest_indices
    return result


def _add_dense_hotspot_neighbor_pairs(
    matrix: np.ndarray,
    base_pairs: List[Tuple[int, int, float]],
    base_pair_keys: set,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> int:
    distances = _pair_distance_matrix(coordinates)
    added_count = 0

    for row, col, base_value in base_pairs:
        if added_count >= DENSE_HOTSPOTS_NEIGHBOR_PAIR_LIMIT:
            break

        row_neighbors = _nearest_stop_indices(
            row,
            distances,
            {row, col},
            DENSE_HOTSPOTS_NEIGHBORS_PER_ENDPOINT,
        )
        col_neighbors = _nearest_stop_indices(
            col,
            distances,
            {row, col},
            DENSE_HOTSPOTS_NEIGHBORS_PER_ENDPOINT,
        )
        candidate_pairs: List[Tuple[int, int]] = []

        for neighbor in row_neighbors:
            candidate_pairs.append((min(neighbor, col), max(neighbor, col)))
        for neighbor in col_neighbors:
            candidate_pairs.append((min(row, neighbor), max(row, neighbor)))
        for row_neighbor, col_neighbor in zip(row_neighbors, col_neighbors):
            candidate_pairs.append((min(row_neighbor, col_neighbor), max(row_neighbor, col_neighbor)))

        for candidate_row, candidate_col in candidate_pairs:
            if added_count >= DENSE_HOTSPOTS_NEIGHBOR_PAIR_LIMIT:
                break
            if candidate_row == candidate_col or (candidate_row, candidate_col) in base_pair_keys:
                continue
            if matrix[candidate_row, candidate_col] > DENSE_HOTSPOTS_BACKGROUND_DEMAND:
                continue

            neighbor_value = min(
                6.0,
                max(1.2, 1.5 + 0.18 * base_value + float(rng.uniform(0.0, 0.8))),
            )
            matrix[candidate_row, candidate_col] = neighbor_value
            matrix[candidate_col, candidate_row] = neighbor_value
            added_count += 1

    result = added_count
    return result


def generate_dense_hotspots_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    node_count = base_matrix.shape[0]
    result = np.full((node_count, node_count), DENSE_HOTSPOTS_BACKGROUND_DEMAND, dtype=float)
    result = zero_diagonal(result)
    base_pairs = _top_base_pairs(base_matrix, node_count * (node_count - 1) // 2)
    base_pair_keys = {(row, col) for row, col, _ in base_pairs}

    for row, col, base_value in base_pairs:
        result[row, col] = _amplify_dense_hotspot_base_value(base_value)
        result[col, row] = result[row, col]

    neighbor_pair_count = _add_dense_hotspot_neighbor_pairs(
        result,
        base_pairs[:12],
        base_pair_keys,
        coordinates,
        rng,
    )

    result = symmetrize(result)
    strong_pair_count = count_pairs_above_background(
        result,
        DENSE_HOTSPOTS_BACKGROUND_DEMAND,
    )
    high_pair_count = int(np.count_nonzero(result[np.triu_indices_from(result, k=1)] > 10.0))
    diagnostics = {
        "background_demand": DENSE_HOTSPOTS_BACKGROUND_DEMAND,
        "strong_pair_count": strong_pair_count,
        "high_pair_count": high_pair_count,
        "preserved_base_pair_count": len(base_pairs),
        "neighbor_pair_count": neighbor_pair_count,
        "top_base_pair_count": len(base_pairs),
        "average_demand": float(result[np.triu_indices_from(result, k=1)].mean()),
    }
    mass_before = off_diagonal_mass(result)
    return result, mass_before, diagnostics


def count_pairs_above_background(matrix: np.ndarray, background_value: float) -> int:
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    result = int(np.count_nonzero(upper > background_value + 1e-9))
    return result


def validate_dense_hotspots_matrix(
    matrix: np.ndarray,
    template_df: pd.DataFrame,
    background_value: float,
) -> Dict[str, object]:
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    positive_values = upper[upper > 0.0]
    row_labels_match = list(template_df.index) == list(template_df.columns)
    validation = {
        "square": matrix.shape[0] == matrix.shape[1],
        "row_column_labels_match": row_labels_match,
        "zero_diagonal": bool(np.allclose(np.diag(matrix), 0.0)),
        "symmetric": bool(np.allclose(matrix, matrix.T)),
        "all_off_diagonal_at_least_background": bool(np.all(upper >= background_value)),
        "strong_pair_count": count_pairs_above_background(matrix, background_value),
        "high_pair_count": int(np.count_nonzero(upper > 10.0)),
        "min_positive": float(positive_values.min()) if len(positive_values) else 0.0,
        "max": float(matrix.max()),
        "mean": float(upper.mean()) if len(upper) else 0.0,
    }
    return validation


def generate_feeder_core_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    total_demand = float(base_matrix.sum())
    center_xy = coordinates.mean(axis=0)
    offsets = coordinates - center_xy
    distances = np.linalg.norm(offsets, axis=1)
    node_count = len(distances)
    max_distance = max(float(distances.max()), 1e-6)

    central_candidate_count = max(
        ACTIVE_CORE_STOP_COUNT + C_SECTOR_COUNT,
        int(math.ceil(node_count * CENTRAL_CANDIDATE_FRACTION)),
    )
    ranked_indices = np.argsort(distances)
    central_candidates = [int(index) for index in ranked_indices[:central_candidate_count]]

    central_angles = np.arctan2(offsets[central_candidates, 1], offsets[central_candidates, 0])
    central_candidates_sorted = [
        stop for _, stop in sorted(zip(central_angles, central_candidates), key=lambda item: item[0])
    ]
    central_sector_groups = np.array_split(np.array(central_candidates_sorted, dtype=int), C_SECTOR_COUNT)
    sector_sizes = [len(group) for group in central_sector_groups]

    gap_start = 0
    best_gap_size = math.inf
    for start_index in range(C_SECTOR_COUNT):
        current_gap_size = sum(
            sector_sizes[(start_index + offset) % C_SECTOR_COUNT]
            for offset in range(C_GAP_SECTOR_COUNT)
        )
        if current_gap_size < best_gap_size:
            best_gap_size = current_gap_size
            gap_start = start_index

    removed_gap_sectors = [
        (gap_start + offset) % C_SECTOR_COUNT
        for offset in range(C_GAP_SECTOR_COUNT)
    ]
    kept_sector_indices = [
        sector_index
        for sector_index in range(C_SECTOR_COUNT)
        if sector_index not in removed_gap_sectors
    ]

    c_shape_candidates: List[int] = []
    c_shape_sector_groups: List[List[int]] = []
    for sector_index in kept_sector_indices:
        sector_group = [int(index) for index in central_sector_groups[sector_index].tolist()]
        c_shape_sector_groups.append(sector_group)
        c_shape_candidates.extend(sector_group)

    c_shape_candidates = sorted(
        c_shape_candidates,
        key=lambda stop_index: math.atan2(offsets[stop_index][1], offsets[stop_index][0]),
    )
    selected_core_stops: List[int] = []
    if c_shape_candidates:
        anchor_positions = np.linspace(
            0,
            len(c_shape_candidates) - 1,
            num=min(ACTIVE_CORE_STOP_COUNT, len(c_shape_candidates)),
        )
        chosen_candidate_set = set()

        for anchor_position in anchor_positions:
            center_position = int(round(float(anchor_position)))
            search_offsets = [0]
            for offset in range(1, len(c_shape_candidates)):
                search_offsets.extend([offset, -offset])

            selected_candidate = None
            best_candidate_score = -math.inf
            for search_offset in search_offsets:
                candidate_position = center_position + search_offset
                if candidate_position < 0 or candidate_position >= len(c_shape_candidates):
                    continue

                candidate_stop = c_shape_candidates[candidate_position]
                if candidate_stop in chosen_candidate_set:
                    continue

                centrality_score = 1.0 - (distances[candidate_stop] / max_distance)
                position_score = 1.0 - (
                    abs(candidate_position - float(anchor_position)) / max(len(c_shape_candidates), 1)
                )
                candidate_score = (0.65 * centrality_score) + (0.35 * position_score)

                if candidate_score > best_candidate_score:
                    selected_candidate = candidate_stop
                    best_candidate_score = candidate_score

                if selected_candidate is not None and abs(search_offset) >= 2:
                    break

            if selected_candidate is not None:
                selected_core_stops.append(int(selected_candidate))
                chosen_candidate_set.add(int(selected_candidate))

    selected_core_stops = sorted(
        list(dict.fromkeys(selected_core_stops)),
        key=lambda stop_index: math.atan2(offsets[stop_index][1], offsets[stop_index][0]),
    )

    gap_sector_center = (
        gap_start + (C_GAP_SECTOR_COUNT - 1) / 2.0
    ) * (2.0 * math.pi / C_SECTOR_COUNT) - math.pi
    noncentral_indices = [
        int(index)
        for index in range(node_count)
        if index not in set(central_candidates)
    ]
    hotspot_index = -1
    hotspot_score = -math.inf
    target_distance = float(np.quantile(distances, 0.65))

    for candidate_index in noncentral_indices:
        candidate_angle = math.atan2(offsets[candidate_index][1], offsets[candidate_index][0])
        angle_gap = abs(
            math.atan2(
                math.sin(candidate_angle - gap_sector_center),
                math.cos(candidate_angle - gap_sector_center),
            )
        )
        angle_score = max(0.0, math.cos(angle_gap))
        distance_gap = abs(distances[candidate_index] - target_distance)
        distance_score = 1.0 - min(distance_gap / max_distance, 1.0)
        tie_break = 1e-6 * float(rng.uniform(0.0, 1.0))
        candidate_score = (0.6 * angle_score) + (0.4 * distance_score) + tie_break

        if candidate_score > hotspot_score:
            hotspot_index = int(candidate_index)
            hotspot_score = candidate_score

    core_matrix = np.zeros((node_count, node_count), dtype=float)
    hotspot_matrix = np.zeros((node_count, node_count), dtype=float)
    core_pairs: List[Tuple[int, int, float]] = []
    hotspot_pairs: List[Tuple[int, int, float]] = []

    for first_index, stop_i in enumerate(selected_core_stops):
        for neighbor_offset in range(1, CORE_NEIGHBORS_PER_STOP + 1):
            second_index = first_index + neighbor_offset
            if second_index >= len(selected_core_stops):
                continue

            stop_j = selected_core_stops[second_index]
            distance_value = float(np.linalg.norm(coordinates[stop_i] - coordinates[stop_j]))
            weight = 1.0 / max(distance_value, 1e-6)
            core_pairs.append((stop_i, stop_j, weight))

    core_weight_sum = sum(weight for _, _, weight in core_pairs)
    core_component_total = total_demand * CORE_SHARE
    if core_weight_sum > 0.0:
        for stop_i, stop_j, weight in core_pairs:
            assigned_value = 0.5 * core_component_total * (weight / core_weight_sum)
            core_matrix[stop_i, stop_j] += assigned_value
            core_matrix[stop_j, stop_i] += assigned_value

    if hotspot_index >= 0 and selected_core_stops:
        hotspot_ranked_core = sorted(
            selected_core_stops,
            key=lambda stop_index: float(np.linalg.norm(coordinates[stop_index] - coordinates[hotspot_index])),
        )
        linked_core_stops = hotspot_ranked_core[: min(HOTSPOT_LINK_COUNT, len(hotspot_ranked_core))]
        hotspot_weight_sum = 0.0

        for stop_index in linked_core_stops:
            distance_value = float(np.linalg.norm(coordinates[stop_index] - coordinates[hotspot_index]))
            weight = 1.0 / max(distance_value, 1e-6)
            hotspot_pairs.append((hotspot_index, stop_index, weight))
            hotspot_weight_sum += weight

        hotspot_component_total = total_demand * HOTSPOT_SHARE
        if hotspot_weight_sum > 0.0:
            for hotspot_stop, core_stop, weight in hotspot_pairs:
                assigned_value = 0.5 * hotspot_component_total * (weight / hotspot_weight_sum)
                hotspot_matrix[hotspot_stop, core_stop] += assigned_value
                hotspot_matrix[core_stop, hotspot_stop] += assigned_value

    result = core_matrix + hotspot_matrix
    result = symmetrize(result)
    result = zero_diagonal(result)
    result = scale_to_total(result, total_demand)
    mass_before = off_diagonal_mass(core_matrix + hotspot_matrix)
    row_demand_sums = result.sum(axis=1)
    ranked_row_stops = np.argsort(row_demand_sums)[::-1][:10]
    top_10_stops_by_row_demand = [
        (int(stop_index), float(row_demand_sums[stop_index]))
        for stop_index in ranked_row_stops
    ]
    maximum_stop_demand_share = (
        float(np.max(row_demand_sums)) / max(float(result.sum()), 1e-6)
    )

    diagnostics = {
        "geometric_center_xy": (float(center_xy[0]), float(center_xy[1])),
        "central_candidate_count": len(central_candidates),
        "C_SECTOR_COUNT": C_SECTOR_COUNT,
        "C_GAP_SECTOR_COUNT": C_GAP_SECTOR_COUNT,
        "removed_gap_sectors": removed_gap_sectors,
        "active_core_stop_indices": selected_core_stops,
        "external_hotspot_stop_index": hotspot_index,
        "CORE_SHARE": CORE_SHARE,
        "HOTSPOT_SHARE": HOTSPOT_SHARE,
        "core_demand_mass": off_diagonal_mass(core_matrix),
        "hotspot_demand_mass": off_diagonal_mass(hotspot_matrix),
        "realized_core_share": off_diagonal_mass(core_matrix) / max(off_diagonal_mass(result), 1e-6),
        "realized_hotspot_share": off_diagonal_mass(hotspot_matrix) / max(off_diagonal_mass(result), 1e-6),
        "core_od_pair_count": nonzero_upper_pairs(core_matrix),
        "hotspot_od_pair_count": nonzero_upper_pairs(hotspot_matrix),
        "final_nonzero_pairs": nonzero_upper_pairs(result),
        "final_density": matrix_density(result),
        "top_10_stops_by_row_demand": top_10_stops_by_row_demand,
        "maximum_stop_demand_share": maximum_stop_demand_share,
    }
    return result, mass_before, diagnostics


def generate_one_center_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    total_demand = float(base_matrix.sum())
    center_xy = coordinates.mean(axis=0)
    offsets = coordinates - center_xy
    distances = np.linalg.norm(offsets, axis=1)
    node_count = len(distances)
    max_distance = max(float(distances.max()), 1e-6)
    stop_angles = np.arctan2(offsets[:, 1], offsets[:, 0])

    center_indices = [
        int(index)
        for index in np.argsort(distances)[:ONE_CENTER_CENTER_STOP_COUNT].tolist()
    ]
    center_indices = sorted(center_indices, key=lambda index: stop_angles[index])
    non_center_indices = [
        int(index)
        for index in range(node_count)
        if int(index) not in set(center_indices)
    ]

    branch_seed_indices: List[int] = []
    if non_center_indices:
        ranked_noncenter = sorted(
            non_center_indices,
            key=lambda index: distances[index],
            reverse=True,
        )
        branch_seed_indices.append(int(ranked_noncenter[0]))

        if len(ranked_noncenter) > 1:
            seed_angle = stop_angles[branch_seed_indices[0]]
            branch_two_index = max(
                ranked_noncenter[1:],
                key=lambda index: (
                    0.6 * (distances[index] / max_distance)
                    + 0.4 * (angle_difference(stop_angles[index], seed_angle) / math.pi)
                ),
            )
            branch_seed_indices.append(int(branch_two_index))

    while len(branch_seed_indices) < EXTERNAL_BRANCH_COUNT and branch_seed_indices:
        branch_seed_indices.append(branch_seed_indices[-1])

    used_indices = set(center_indices)
    branch_stop_indices: List[List[int]] = []
    branch_connected_centers: List[int] = []
    branch_weights: List[List[Tuple[int, int, float]]] = []
    branch_share_values = [BRANCH_1_SHARE, BRANCH_2_SHARE]

    for branch_index in range(min(EXTERNAL_BRANCH_COUNT, len(branch_seed_indices))):
        seed_index = branch_seed_indices[branch_index]
        seed_angle = stop_angles[seed_index]
        ranked_branch_candidates = sorted(
            [
                index
                for index in non_center_indices
                if index not in used_indices
            ],
            key=lambda index: (
                -(
                    0.7 * (distances[index] / max_distance)
                    + 0.3 * (1.0 - min(angle_difference(stop_angles[index], seed_angle) / math.pi, 1.0))
                ),
                distances[index],
            ),
        )

        branch_target_count = max(
            MIN_BRANCH_STOP_COUNT,
            min(MAX_BRANCH_STOP_COUNT, MAX_BRANCH_STOP_COUNT - branch_index),
        )
        selected_branch = [int(index) for index in ranked_branch_candidates[:branch_target_count]]
        selected_branch = sorted(selected_branch, key=lambda index: distances[index], reverse=True)
        used_indices.update(selected_branch)
        branch_stop_indices.append(selected_branch)

        connected_center = min(
            center_indices,
            key=lambda center_index: float(
                np.linalg.norm(coordinates[center_index] - coordinates[selected_branch[-1]])
            ),
        )
        branch_connected_centers.append(int(connected_center))

        current_branch_weights: List[Tuple[int, int, float]] = []
        for local_index in range(len(selected_branch) - 1):
            stop_i = selected_branch[local_index]
            stop_j = selected_branch[local_index + 1]
            weight = 1.0 / max(
                float(np.linalg.norm(coordinates[stop_i] - coordinates[stop_j])),
                1e-6,
            )
            current_branch_weights.append((stop_i, stop_j, weight))

        connector_stop = selected_branch[-1]
        connector_weight = 1.0 / max(
            float(np.linalg.norm(coordinates[connector_stop] - coordinates[connected_center])),
            1e-6,
        )
        current_branch_weights.append((connector_stop, int(connected_center), connector_weight))
        branch_weights.append(current_branch_weights)

    core_matrix = np.zeros((node_count, node_count), dtype=float)
    branch_1_matrix = np.zeros((node_count, node_count), dtype=float)
    branch_2_matrix = np.zeros((node_count, node_count), dtype=float)
    cross_matrix = np.zeros((node_count, node_count), dtype=float)

    core_pairs: List[Tuple[int, int, float]] = []
    for first_index in range(len(center_indices) - 1):
        stop_i = center_indices[first_index]
        stop_j = center_indices[first_index + 1]
        weight = 1.0 / max(
            float(np.linalg.norm(coordinates[stop_i] - coordinates[stop_j])),
            1e-6,
        )
        core_pairs.append((stop_i, stop_j, weight))

    if len(center_indices) >= 4:
        extra_pair = (center_indices[0], center_indices[2])
        extra_weight = 0.6 / max(
            float(np.linalg.norm(coordinates[extra_pair[0]] - coordinates[extra_pair[1]])),
            1e-6,
        )
        core_pairs.append((extra_pair[0], extra_pair[1], extra_weight))

    core_weight_sum = sum(weight for _, _, weight in core_pairs)
    if core_weight_sum > 0.0:
        for stop_i, stop_j, weight in core_pairs:
            assigned_value = 0.5 * total_demand * ONE_CENTER_CORE_SHARE * (weight / core_weight_sum)
            core_matrix[stop_i, stop_j] += assigned_value
            core_matrix[stop_j, stop_i] += assigned_value

    branch_matrices = [branch_1_matrix, branch_2_matrix]
    branch_masses: List[float] = []
    for branch_index, current_branch_weights in enumerate(branch_weights[:2]):
        weight_sum = sum(weight for _, _, weight in current_branch_weights)
        current_mass = total_demand * branch_share_values[branch_index]
        branch_masses.append(current_mass)
        if weight_sum > 0.0:
            for stop_i, stop_j, weight in current_branch_weights:
                assigned_value = 0.5 * current_mass * (weight / weight_sum)
                branch_matrices[branch_index][stop_i, stop_j] += assigned_value
                branch_matrices[branch_index][stop_j, stop_i] += assigned_value

    cross_pairs: List[Tuple[int, int, float]] = []
    for branch_index, selected_branch in enumerate(branch_stop_indices[:2]):
        if not selected_branch:
            continue

        branch_outer_stop = selected_branch[0]
        available_centers = [
            center_index
            for center_index in center_indices
            if center_index != branch_connected_centers[branch_index]
        ]
        if not available_centers:
            continue

        cross_center = min(
            available_centers,
            key=lambda center_index: float(
                np.linalg.norm(coordinates[branch_outer_stop] - coordinates[center_index])
            ),
        )
        cross_weight = 1.0 / max(
            float(np.linalg.norm(coordinates[branch_outer_stop] - coordinates[cross_center])),
            1e-6,
        )
        cross_pairs.append((branch_outer_stop, int(cross_center), cross_weight))

    cross_weight_sum = sum(weight for _, _, weight in cross_pairs)
    if cross_weight_sum > 0.0:
        for stop_i, stop_j, weight in cross_pairs:
            assigned_value = 0.5 * total_demand * CROSS_SHARE * (weight / cross_weight_sum)
            cross_matrix[stop_i, stop_j] += assigned_value
            cross_matrix[stop_j, stop_i] += assigned_value

    result = core_matrix + branch_1_matrix + branch_2_matrix + cross_matrix
    result = symmetrize(result)
    result = zero_diagonal(result)
    result = scale_to_total(result, total_demand)
    mass_before = off_diagonal_mass(core_matrix + branch_1_matrix + branch_2_matrix + cross_matrix)
    row_demand_sums = result.sum(axis=1)
    ranked_row_stops = np.argsort(row_demand_sums)[::-1][:10]
    top_10_stops_by_row_demand = [
        (int(stop_index), float(row_demand_sums[stop_index]))
        for stop_index in ranked_row_stops
    ]
    maximum_stop_demand_share = float(np.max(row_demand_sums)) / max(float(result.sum()), 1e-6)

    diagnostics = {
        "center_coordinates": (float(center_xy[0]), float(center_xy[1])),
        "central_stop_indices": center_indices,
        "external_branch_stop_indices": branch_stop_indices,
        "branch_connected_center_indices": branch_connected_centers,
        "CORE_SHARE": ONE_CENTER_CORE_SHARE,
        "BRANCH_1_SHARE": BRANCH_1_SHARE,
        "BRANCH_2_SHARE": BRANCH_2_SHARE,
        "CROSS_SHARE": CROSS_SHARE,
        "core_mass": off_diagonal_mass(core_matrix),
        "branch_1_mass": off_diagonal_mass(branch_1_matrix),
        "branch_2_mass": off_diagonal_mass(branch_2_matrix),
        "cross_mass": off_diagonal_mass(cross_matrix),
        "total_mass": off_diagonal_mass(result),
        "final_nonzero_pairs": nonzero_upper_pairs(result),
        "final_density": matrix_density(result),
        "top_10_stops_by_row_demand": top_10_stops_by_row_demand,
        "maximum_stop_demand_share": maximum_stop_demand_share,
    }
    return result, mass_before, diagnostics


def generate_two_centers_separated_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    total_demand = float(base_matrix.sum())
    center_i, center_j = select_two_centers(coordinates)
    affinity_i = gaussian_affinity(coordinates, center_i)
    affinity_j = gaussian_affinity(coordinates, center_j)
    labels = np.where(affinity_i >= affinity_j, 0, 1)

    dense_matrix = np.outer(affinity_i, affinity_i) + np.outer(affinity_j, affinity_j)
    dense_matrix = symmetrize(dense_matrix)
    same_cluster_mask = labels[:, None] == labels[None, :]
    dense_matrix = np.where(same_cluster_mask, dense_matrix, 0.0)
    dense_matrix = zero_diagonal(dense_matrix)
    mass_before = off_diagonal_mass(dense_matrix)
    sparse_matrix = sparsify_top_pairs(dense_matrix, TOP_FRACTION, rng)
    result = scale_to_total(sparse_matrix, total_demand)
    return result, mass_before


def generate_two_centers_bridge_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    total_demand = float(base_matrix.sum())
    center_a_index, center_b_index = select_two_centers(coordinates)
    affinity_a = gaussian_affinity(
        coordinates,
        center_a_index,
        sigma_scale=SIGMA_A_FACTOR,
    )
    affinity_b = gaussian_affinity(
        coordinates,
        center_b_index,
        sigma_scale=SIGMA_B_FACTOR,
    )
    labels = np.where(affinity_a >= affinity_b, 0, 1)

    cluster_a_dense = np.outer(affinity_a, affinity_a)
    cluster_b_dense = np.outer(affinity_b, affinity_b)
    cluster_a_mask = (labels[:, None] == 0) & (labels[None, :] == 0)
    cluster_b_mask = (labels[:, None] == 1) & (labels[None, :] == 1)

    cluster_a_dense = np.where(cluster_a_mask, cluster_a_dense, 0.0)
    cluster_b_dense = np.where(cluster_b_mask, cluster_b_dense, 0.0)
    cluster_a_dense = zero_diagonal(symmetrize(cluster_a_dense))
    cluster_b_dense = zero_diagonal(symmetrize(cluster_b_dense))
    mass_before = off_diagonal_mass(cluster_a_dense + cluster_b_dense)

    cluster_a_sparse = sparsify_top_pairs_by_mask(
        cluster_a_dense,
        cluster_a_mask,
        CLUSTER_A_TOP_FRACTION,
        rng,
    )
    cluster_b_sparse = sparsify_top_pairs_by_mask(
        cluster_b_dense,
        cluster_b_mask,
        CLUSTER_B_TOP_FRACTION,
        rng,
    )
    cluster_a_sparse = scale_to_total(cluster_a_sparse, total_demand * CLUSTER_A_SHARE)
    cluster_b_sparse = scale_to_total(cluster_b_sparse, total_demand * CLUSTER_B_SHARE)

    bridge_dense = np.outer(affinity_a, affinity_b) + np.outer(affinity_b, affinity_a)
    bridge_dense = symmetrize(bridge_dense)
    cross_center_mask = labels[:, None] != labels[None, :]
    bridge_sparse = np.zeros_like(bridge_dense, dtype=float)
    selected_bridge_pair_indices = (-1, -1)
    selected_bridge_score = 0.0

    for row in range(bridge_dense.shape[0]):
        for col in range(row + 1, bridge_dense.shape[1]):
            bridge_score = float(bridge_dense[row, col])
            if cross_center_mask[row, col] and bridge_score > selected_bridge_score:
                selected_bridge_pair_indices = (row, col)
                selected_bridge_score = bridge_score

    if selected_bridge_pair_indices[0] >= 0:
        row, col = selected_bridge_pair_indices
        bridge_sparse[row, col] = selected_bridge_score
        bridge_sparse[col, row] = selected_bridge_score

    bridge_sparse = scale_to_total(bridge_sparse, total_demand * BRIDGE_SHARE)

    result = cluster_a_sparse + cluster_b_sparse + bridge_sparse
    result = symmetrize(result)
    result = zero_diagonal(result)
    result = scale_to_total(result, total_demand)

    selected_bridge_pair_stop_ids = None
    selected_bridge_pair_demand = 0.0
    if selected_bridge_pair_indices[0] >= 0:
        row, col = selected_bridge_pair_indices
        selected_bridge_pair_stop_ids = (row, col)
        selected_bridge_pair_demand = float(result[row, col])

    diagnostics = {
        "center_a_index": center_a_index,
        "center_b_index": center_b_index,
        "center_a_stop_id": center_a_index,
        "center_b_stop_id": center_b_index,
        "CLUSTER_A_SHARE": CLUSTER_A_SHARE,
        "CLUSTER_B_SHARE": CLUSTER_B_SHARE,
        "BRIDGE_SHARE": BRIDGE_SHARE,
        "BRIDGE_PAIR_COUNT": BRIDGE_PAIR_COUNT,
        "cluster_a_demand_mass": off_diagonal_mass(cluster_a_sparse),
        "cluster_b_demand_mass": off_diagonal_mass(cluster_b_sparse),
        "total_demand_mass": off_diagonal_mass(result),
        "realized_cluster_a_share": (
            off_diagonal_mass(cluster_a_sparse) / off_diagonal_mass(result)
            if off_diagonal_mass(result) > 0.0
            else 0.0
        ),
        "realized_cluster_b_share": (
            off_diagonal_mass(cluster_b_sparse) / off_diagonal_mass(result)
            if off_diagonal_mass(result) > 0.0
            else 0.0
        ),
        "selected_bridge_pair_indices": selected_bridge_pair_indices,
        "selected_bridge_pair_stop_ids": selected_bridge_pair_stop_ids,
        "selected_bridge_pair_demand": selected_bridge_pair_demand,
        "intra_demand_mass": off_diagonal_mass(cluster_a_sparse + cluster_b_sparse),
        "bridge_demand_mass": off_diagonal_mass(bridge_sparse),
        "realized_bridge_share": (
            off_diagonal_mass(bridge_sparse) / off_diagonal_mass(result)
            if off_diagonal_mass(result) > 0.0
            else 0.0
        ),
        "cluster_a_nonzero_pairs": nonzero_upper_pairs(cluster_a_sparse),
        "cluster_b_nonzero_pairs": nonzero_upper_pairs(cluster_b_sparse),
        "nonzero_bridge_pairs": nonzero_upper_pairs(bridge_sparse),
        "cluster_a_stop_count": int(np.count_nonzero(labels == 0)),
        "cluster_b_stop_count": int(np.count_nonzero(labels == 1)),
    }
    return result, mass_before, diagnostics


def generate_corridor_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    total_demand = float(base_matrix.sum())
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    axis, orth_axis = principal_axis(coordinates)
    longitudinal = centered @ axis
    lateral = centered @ orth_axis

    lateral_sigma = max(np.std(lateral) * 0.60, 1e-6)
    longitudinal_sigma = max(np.std(longitudinal) * 0.90, 1e-6)
    corridor_weight = np.exp(-(lateral ** 2) / (2.0 * lateral_sigma ** 2))
    longitudinal_gap = np.abs(longitudinal[:, None] - longitudinal[None, :])
    continuity = np.exp(-longitudinal_gap / longitudinal_sigma)

    dense_matrix = np.outer(corridor_weight, corridor_weight) * continuity
    dense_matrix = symmetrize(dense_matrix)
    mass_before = off_diagonal_mass(dense_matrix)
    sparse_matrix = sparsify_top_pairs(dense_matrix, TOP_FRACTION, rng)
    result = scale_to_total(sparse_matrix, total_demand)
    return result, mass_before


def generate_center_periphery_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    total_demand = float(base_matrix.sum())
    center_xy = coordinates.mean(axis=0)
    offsets = coordinates - center_xy
    distances = np.linalg.norm(offsets, axis=1)
    node_count = len(distances)

    ranked_indices = np.argsort(distances)
    center_indices = ranked_indices[:CENTER_STOP_COUNT]

    peripheral_candidate_count = max(
        SECTOR_COUNT * MIN_ACTIVE_PERIPHERAL_STOPS_PER_SECTOR,
        math.ceil(node_count * PERIPHERY_FRACTION),
    )
    peripheral_candidates = [
        int(index)
        for index in ranked_indices[::-1]
        if int(index) not in set(center_indices)
    ][:peripheral_candidate_count]

    candidate_offsets = offsets[peripheral_candidates]
    candidate_angles = np.arctan2(candidate_offsets[:, 1], candidate_offsets[:, 0])
    ordered_candidates = [
        stop for _, stop in sorted(zip(candidate_angles, peripheral_candidates), key=lambda item: item[0])
    ]
    sector_groups = np.array_split(np.array(ordered_candidates, dtype=int), SECTOR_COUNT)

    center_angles = {
        int(index): math.atan2(offsets[index][1], offsets[index][0]) if distances[index] > 0.0 else 0.0
        for index in center_indices
    }
    radial_matrix = np.zeros((node_count, node_count), dtype=float)
    core_matrix = np.zeros((node_count, node_count), dtype=float)
    local_peripheral_matrix = np.zeros((node_count, node_count), dtype=float)
    sector_stop_counts: List[int] = []
    active_peripheral_indices_per_sector: List[List[int]] = []
    nonzero_radial_pairs = 0
    nonzero_local_pairs = 0

    for sector_index, sector_group in enumerate(sector_groups):
        sector_list = [int(index) for index in sector_group.tolist()]
        sector_stop_counts.append(len(sector_list))

        if not sector_list:
            active_peripheral_indices_per_sector.append([])
            continue

        sector_angles = np.arctan2(offsets[sector_list, 1], offsets[sector_list, 0])
        sector_mean_angle = math.atan2(
            float(np.mean(np.sin(sector_angles))),
            float(np.mean(np.cos(sector_angles))),
        )

        scored_sector_stops: List[Tuple[int, float]] = []
        for stop_index in sector_list:
            stop_angle = math.atan2(offsets[stop_index][1], offsets[stop_index][0])
            angle_gap = abs(math.atan2(math.sin(stop_angle - sector_mean_angle), math.cos(stop_angle - sector_mean_angle)))
            alignment = max(0.0, math.cos(angle_gap))
            distance_score = distances[stop_index] / max(float(distances.max()), 1e-6)
            tie_break = 1e-6 * float(rng.uniform(0.0, 1.0))
            score = (0.7 * distance_score) + (0.3 * alignment) + tie_break
            scored_sector_stops.append((stop_index, score))

        scored_sector_stops.sort(key=lambda item: item[1], reverse=True)
        active_count = math.ceil(len(scored_sector_stops) * ACTIVE_PERIPHERAL_FRACTION)
        active_count = max(MIN_ACTIVE_PERIPHERAL_STOPS_PER_SECTOR, active_count)
        active_count = min(MAX_ACTIVE_PERIPHERAL_STOPS_PER_SECTOR, active_count, len(scored_sector_stops))
        active_stops = [stop for stop, _ in scored_sector_stops[:active_count]]
        active_peripheral_indices_per_sector.append(active_stops)

        radial_weights: List[Tuple[int, int, float]] = []
        for local_index, stop_index in enumerate(active_stops):
            stop_angle = math.atan2(offsets[stop_index][1], offsets[stop_index][0])
            ranked_centers = sorted(
                center_indices,
                key=lambda center_index: abs(
                    math.atan2(
                        math.sin(stop_angle - center_angles[int(center_index)]),
                        math.cos(stop_angle - center_angles[int(center_index)]),
                    )
                ),
            )
            linked_centers = [int(ranked_centers[local_index % len(ranked_centers)])][:CENTER_LINKS_PER_PERIPHERAL_STOP]
            stop_weight = scored_sector_stops[local_index][1]

            for center_index in linked_centers:
                radial_weights.append((center_index, stop_index, stop_weight))

        radial_weight_sum = sum(weight for _, _, weight in radial_weights)
        sector_radial_total = total_demand * RADIAL_DEMAND_SHARE * SECTOR_SHARES[sector_index]
        if radial_weight_sum > 0.0:
            for center_index, stop_index, stop_weight in radial_weights:
                assigned_value = 0.5 * sector_radial_total * (stop_weight / radial_weight_sum)
                radial_matrix[center_index, stop_index] += assigned_value
                radial_matrix[stop_index, center_index] += assigned_value
            nonzero_radial_pairs += len(radial_weights)

        local_pairs: List[Tuple[int, int, float]] = []
        for local_index in range(len(active_stops) - 1):
            stop_i = active_stops[local_index]
            stop_j = active_stops[local_index + 1]
            closeness = 1.0 / (1.0 + float(np.linalg.norm(coordinates[stop_i] - coordinates[stop_j])))
            local_pairs.append((stop_i, stop_j, closeness))

        local_weight_sum = sum(weight for _, _, weight in local_pairs)
        sector_local_total = total_demand * LOCAL_PERIPHERAL_SHARE * SECTOR_SHARES[sector_index]
        if local_weight_sum > 0.0:
            for stop_i, stop_j, local_weight in local_pairs:
                assigned_value = 0.5 * sector_local_total * (local_weight / local_weight_sum)
                local_peripheral_matrix[stop_i, stop_j] += assigned_value
                local_peripheral_matrix[stop_j, stop_i] += assigned_value
            nonzero_local_pairs += len(local_pairs)

    core_pairs: List[Tuple[int, int]] = []
    for first_index in range(len(center_indices)):
        for second_index in range(first_index + 1, len(center_indices)):
            stop_i = int(center_indices[first_index])
            stop_j = int(center_indices[second_index])
            core_pairs.append((stop_i, stop_j))

    if core_pairs:
        core_pair_value = (total_demand * CORE_INTERNAL_SHARE) / (2.0 * len(core_pairs))
        for stop_i, stop_j in core_pairs:
            core_matrix[stop_i, stop_j] += core_pair_value
            core_matrix[stop_j, stop_i] += core_pair_value

    result = radial_matrix + core_matrix + local_peripheral_matrix
    result = symmetrize(result)
    result = zero_diagonal(result)
    result = scale_to_total(result, total_demand)
    mass_before = off_diagonal_mass(radial_matrix + core_matrix + local_peripheral_matrix)
    row_demand_sums = result.sum(axis=1)
    ranked_row_stops = np.argsort(row_demand_sums)[::-1][:10]
    top_10_stops_by_row_demand = [
        (int(stop_index), float(row_demand_sums[stop_index]))
        for stop_index in ranked_row_stops
    ]
    maximum_stop_demand_share = (
        float(np.max(row_demand_sums)) / max(float(result.sum()), 1e-6)
    )

    diagnostics = {
        "center_stop_indices": [int(index) for index in center_indices.tolist()],
        "peripheral_candidate_count": len(peripheral_candidates),
        "sector_count": SECTOR_COUNT,
        "sector_stop_counts": sector_stop_counts,
        "active_peripheral_indices_per_sector": active_peripheral_indices_per_sector,
        "SECTOR_SHARES": list(SECTOR_SHARES),
        "RADIAL_DEMAND_SHARE": RADIAL_DEMAND_SHARE,
        "CORE_INTERNAL_SHARE": CORE_INTERNAL_SHARE,
        "LOCAL_PERIPHERAL_SHARE": LOCAL_PERIPHERAL_SHARE,
        "radial_mass": off_diagonal_mass(radial_matrix),
        "core_internal_mass": off_diagonal_mass(core_matrix),
        "local_peripheral_mass": off_diagonal_mass(local_peripheral_matrix),
        "total_mass": off_diagonal_mass(result),
        "realized_radial_share": off_diagonal_mass(radial_matrix) / max(off_diagonal_mass(result), 1e-6),
        "realized_core_internal_share": off_diagonal_mass(core_matrix) / max(off_diagonal_mass(result), 1e-6),
        "realized_local_peripheral_share": off_diagonal_mass(local_peripheral_matrix) / max(off_diagonal_mass(result), 1e-6),
        "nonzero_radial_pairs": nonzero_radial_pairs,
        "nonzero_core_pairs": len(core_pairs),
        "nonzero_local_peripheral_pairs": nonzero_local_pairs,
        "final_nonzero_pairs": nonzero_upper_pairs(result),
        "final_density": matrix_density(result),
        "top_10_stops_by_row_demand": top_10_stops_by_row_demand,
        "maximum_stop_demand_share": maximum_stop_demand_share,
    }
    return result, mass_before, diagnostics


def write_scenario(
    name: str,
    matrix: np.ndarray,
    template_df: pd.DataFrame,
    scenario_dir: str,
    total_demand: float,
    mass_before_sparsification: float,
) -> str:
    output_path = os.path.join(scenario_dir, f"od_{name}.csv")
    scenario_df = matrix_to_dataframe(matrix, template_df)
    scenario_df.to_csv(output_path)
    print(f"  -> escrit: {output_path}")
    return output_path


PairDemand = Dict[Tuple[int, int], float]


def add_balanced_pair(pairs: PairDemand, stop_a: int, stop_b: int, value: float) -> None:
    if stop_a != stop_b:
        key = (stop_a, stop_b) if stop_a < stop_b else (stop_b, stop_a)
        pairs[key] = max(float(value), pairs.get(key, 0.0))


def matrix_from_balanced_pairs(
    pairs: PairDemand,
    node_count: int,
    background_demand: float = BALANCED_BACKGROUND_DEMAND,
) -> np.ndarray:
    matrix = np.full((node_count, node_count), float(background_demand), dtype=float)
    np.fill_diagonal(matrix, 0.0)

    for (stop_a, stop_b), value in pairs.items():
        matrix[stop_a, stop_b] = value
        matrix[stop_b, stop_a] = value

    return zero_diagonal(matrix)


def nearest_stop_indices(
    coordinates: np.ndarray,
    anchor_index: int,
    count: int,
    excluded: List[int] | None = None,
) -> List[int]:
    excluded_set = set(excluded or [])
    distances = np.linalg.norm(coordinates - coordinates[anchor_index], axis=1)
    ranked = np.argsort(distances)
    selected = [
        int(index)
        for index in ranked
        if int(index) not in excluded_set
    ][:count]
    return selected


def extreme_stop_index(coordinates: np.ndarray, direction: Tuple[float, float]) -> int:
    direction_array = np.array(direction, dtype=float)
    scores = coordinates @ direction_array
    return int(np.argmax(scores))


def add_chain_pairs(
    pairs: PairDemand,
    ordered_indices: List[int],
    value: float,
    skip_value: float | None = None,
) -> None:
    for index in range(len(ordered_indices) - 1):
        add_balanced_pair(pairs, ordered_indices[index], ordered_indices[index + 1], value)

    if skip_value is not None:
        for index in range(len(ordered_indices) - 2):
            add_balanced_pair(pairs, ordered_indices[index], ordered_indices[index + 2], skip_value)


def add_compact_internal_pairs(
    pairs: PairDemand,
    group: List[int],
    coordinates: np.ndarray,
    pair_count: int,
    strong_value: float,
    medium_value: float,
) -> None:
    candidate_pairs: List[Tuple[float, int, int]] = []

    for row_index in range(len(group)):
        for col_index in range(row_index + 1, len(group)):
            stop_a = group[row_index]
            stop_b = group[col_index]
            distance = float(np.linalg.norm(coordinates[stop_a] - coordinates[stop_b]))
            candidate_pairs.append((distance, stop_a, stop_b))

    candidate_pairs.sort(key=lambda item: item[0])

    for rank, (_, stop_a, stop_b) in enumerate(candidate_pairs[:pair_count]):
        value = strong_value if rank < max(2, pair_count // 2) else medium_value
        add_balanced_pair(pairs, stop_a, stop_b, value)


def add_ranked_group_pairs(
    pairs: PairDemand,
    group_a: List[int],
    group_b: List[int],
    coordinates: np.ndarray,
    pair_count: int,
    values: List[float],
    same_group: bool = False,
) -> None:
    candidate_pairs: List[Tuple[float, int, int]] = []

    for index_a, stop_a in enumerate(group_a):
        for index_b, stop_b in enumerate(group_b):
            if stop_a == stop_b:
                continue
            if same_group and index_b <= index_a:
                continue
            distance = float(np.linalg.norm(coordinates[stop_a] - coordinates[stop_b]))
            candidate_pairs.append((distance, stop_a, stop_b))

    candidate_pairs.sort(key=lambda item: item[0])

    for rank, (_, stop_a, stop_b) in enumerate(candidate_pairs[:pair_count]):
        value = values[min(rank, len(values) - 1)]
        add_balanced_pair(pairs, stop_a, stop_b, value)


def add_axis_corridor_pairs(
    pairs: PairDemand,
    coordinates: np.ndarray,
    axis: int,
    pair_count: int,
    value: float,
    step: int = 2,
) -> None:
    ordered = [int(index) for index in np.argsort(coordinates[:, axis])]
    added = 0

    for offset in range(step):
        for index in range(offset, len(ordered) - step, step + 1):
            if added >= pair_count:
                break
            add_balanced_pair(pairs, ordered[index], ordered[index + step], value)
            added += 1
        if added >= pair_count:
            break


def relevant_pair_count(matrix: np.ndarray) -> int:
    return nonzero_upper_pairs(matrix)


def print_balanced_summary(name: str, matrix: np.ndarray, template_df: pd.DataFrame) -> bool:
    validation = validate_matrix(matrix)
    same_labels = bool(template_df.index.equals(template_df.columns))
    upper_values = matrix[np.triu_indices_from(matrix, k=1)]
    pair_count = int(len(upper_values))
    pairs_above_background = int(np.count_nonzero(upper_values > BALANCED_BACKGROUND_DEMAND))
    pairs_above_one = int(np.count_nonzero(upper_values > 1.0))
    pairs_above_three = int(np.count_nonzero(upper_values > 3.0))
    pairs_above_ten = int(np.count_nonzero(upper_values > 10.0))
    mean_unordered = float(upper_values.mean()) if len(upper_values) else 0.0
    total_unordered = float(upper_values.sum())
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    off_diagonal_min = float(off_diagonal.min()) if len(off_diagonal) else 0.0
    passed = bool(
        validation["shape"][0] == validation["shape"][1]
        and validation["symmetric"]
        and validation["zero_diagonal"]
        and same_labels
        and off_diagonal_min >= BALANCED_BACKGROUND_DEMAND
        and 80 <= pairs_above_background <= 120
    )

    print(f"\n{name}:")
    print(f" - number_of_stops = {matrix.shape[0]}")
    print(f" - unordered_OD_pairs = {pair_count}")
    print(f" - pairs_above_0p1 = {pairs_above_background}")
    print(f" - pairs_above_1p0 = {pairs_above_one}")
    print(f" - pairs_above_3p0 = {pairs_above_three}")
    print(f" - pairs_above_10p0 = {pairs_above_ten}")
    print(f" - max_demand = {validation['max']:.6f}")
    print(f" - mean_unordered_demand = {mean_unordered:.6f}")
    print(f" - total_unordered_demand = {total_unordered:.6f}")
    print(f" - min_off_diagonal_demand = {off_diagonal_min:.6f}")
    print(f" - symmetric = {validation['symmetric']}")
    print(f" - zero_diagonal = {validation['zero_diagonal']}")
    print(f" - labels_match = {same_labels}")
    print(f" - validation_passed = {passed}")
    return passed


def generate_balanced_base_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    target_total: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pairs: PairDemand = {}
    original_pairs = _top_base_pairs(base_matrix, limit=18)

    for rank, (stop_a, stop_b, demand) in enumerate(original_pairs):
        if demand >= 5.0:
            value = 18.0
        elif demand >= 2.0:
            value = 10.0
        elif rank < 12:
            value = 5.5
        else:
            value = 2.2
        add_balanced_pair(pairs, stop_a, stop_b, value)

    anchors = [
        extreme_stop_index(coordinates, (-1.0, 0.0)),
        extreme_stop_index(coordinates, (1.0, 0.0)),
        extreme_stop_index(coordinates, (0.0, -1.0)),
        extreme_stop_index(coordinates, (0.0, 1.0)),
        geometric_center_index(coordinates),
    ]
    for anchor in anchors:
        local_group = nearest_stop_indices(coordinates, anchor, 3)
        add_chain_pairs(pairs, local_group, value=2.2, skip_value=1.2)

    for stop_a in rng.choice(np.arange(coordinates.shape[0]), size=2, replace=False):
        stop_b = nearest_stop_indices(coordinates, int(stop_a), 4)[-1]
        add_balanced_pair(pairs, int(stop_a), stop_b, 1.0)

    left_group = nearest_stop_indices(coordinates, extreme_stop_index(coordinates, (-1.0, 0.0)), 9)
    right_group = nearest_stop_indices(coordinates, extreme_stop_index(coordinates, (1.0, 0.0)), 9)
    top_group = nearest_stop_indices(coordinates, extreme_stop_index(coordinates, (0.0, 1.0)), 9)
    bottom_group = nearest_stop_indices(coordinates, extreme_stop_index(coordinates, (0.0, -1.0)), 9)
    center_group = nearest_stop_indices(coordinates, geometric_center_index(coordinates), 10)

    medium_values = [6.5] * 4 + [2.4] * 8
    for group in [left_group, right_group, top_group, bottom_group, center_group]:
        add_ranked_group_pairs(
            pairs,
            group,
            group,
            coordinates,
            pair_count=12,
            values=medium_values,
            same_group=True,
        )

    add_ranked_group_pairs(
        pairs,
        left_group,
        right_group,
        coordinates,
        pair_count=8,
        values=[13.0] * 4 + [6.0] * 4,
    )
    add_ranked_group_pairs(
        pairs,
        top_group,
        bottom_group,
        coordinates,
        pair_count=8,
        values=[11.5] * 3 + [5.0] * 5,
    )
    add_axis_corridor_pairs(pairs, coordinates, axis=0, pair_count=12, value=2.4, step=3)
    add_axis_corridor_pairs(pairs, coordinates, axis=1, pair_count=10, value=2.2, step=3)

    matrix = matrix_from_balanced_pairs(pairs, base_matrix.shape[0])
    return matrix


def generate_balanced_one_center_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    target_total: float,
    rng: np.random.Generator,
) -> np.ndarray:
    del rng
    pairs: PairDemand = {}
    center_anchor = geometric_center_index(coordinates)
    center_group = nearest_stop_indices(coordinates, center_anchor, 10)
    near_outer = nearest_stop_indices(coordinates, center_anchor, 24, excluded=center_group)[0:14]
    medium_outer = nearest_stop_indices(
        coordinates,
        center_anchor,
        40,
        excluded=center_group + near_outer,
    )[0:16]

    add_compact_internal_pairs(
        pairs,
        center_group,
        coordinates,
        pair_count=28,
        strong_value=18.0,
        medium_value=8.0,
    )

    for index, outer_stop in enumerate(near_outer):
        center_stop = center_group[index % len(center_group)]
        second_center_stop = center_group[(index + 3) % len(center_group)]
        add_balanced_pair(pairs, center_stop, outer_stop, 12.0 if index < 6 else 6.5)
        add_balanced_pair(pairs, second_center_stop, outer_stop, 4.2 if index < 8 else 2.4)

    for index, outer_stop in enumerate(medium_outer):
        center_stop = center_group[(index * 2) % len(center_group)]
        add_balanced_pair(pairs, center_stop, outer_stop, 3.6 if index < 10 else 1.6)

    sorted_outer = sorted(near_outer + medium_outer[:8], key=lambda item: coordinates[item][0])
    add_chain_pairs(pairs, sorted_outer[:22], value=1.6)
    add_ranked_group_pairs(
        pairs,
        near_outer,
        near_outer,
        coordinates,
        pair_count=18,
        values=[4.5] * 6 + [2.2] * 12,
        same_group=True,
    )

    matrix = matrix_from_balanced_pairs(pairs, base_matrix.shape[0])
    return matrix


def generate_balanced_two_centers_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    target_total: float,
    rng: np.random.Generator,
) -> np.ndarray:
    del rng
    pairs: PairDemand = {}
    left_anchor = extreme_stop_index(coordinates, (-1.0, 0.0))
    right_anchor = extreme_stop_index(coordinates, (1.0, 0.0))
    group_a = nearest_stop_indices(coordinates, left_anchor, 10)
    group_b = nearest_stop_indices(coordinates, right_anchor, 10, excluded=group_a)
    periphery_a = nearest_stop_indices(coordinates, left_anchor, 22, excluded=group_a + group_b)[0:10]
    periphery_b = nearest_stop_indices(coordinates, right_anchor, 22, excluded=group_a + group_b + periphery_a)[0:10]

    add_compact_internal_pairs(pairs, group_a, coordinates, 18, 12.0, 7.0)
    add_compact_internal_pairs(pairs, group_b, coordinates, 18, 12.0, 7.0)

    group_a_reps = sorted(group_a, key=lambda item: coordinates[item][0], reverse=True)[:7]
    group_b_reps = sorted(group_b, key=lambda item: coordinates[item][0])[:7]
    add_ranked_group_pairs(
        pairs,
        group_a_reps,
        group_b_reps,
        coordinates,
        pair_count=15,
        values=[12.0] * 2 + [7.0] * 6 + [3.5] * 7,
    )
    add_ranked_group_pairs(
        pairs,
        group_a,
        periphery_a,
        coordinates,
        pair_count=10,
        values=[5.0] * 6 + [2.2] * 4,
    )
    add_ranked_group_pairs(
        pairs,
        group_b,
        periphery_b,
        coordinates,
        pair_count=10,
        values=[5.0] * 6 + [2.2] * 4,
    )
    add_chain_pairs(pairs, periphery_a, value=1.6)
    add_chain_pairs(pairs, periphery_b, value=1.6)

    matrix = matrix_from_balanced_pairs(pairs, base_matrix.shape[0])
    return matrix


def generate_balanced_center_periphery_matrix(
    base_matrix: np.ndarray,
    coordinates: np.ndarray,
    target_total: float,
    rng: np.random.Generator,
) -> np.ndarray:
    del rng
    pairs: PairDemand = {}
    center_anchor = geometric_center_index(coordinates)
    center_group = nearest_stop_indices(coordinates, center_anchor, 8)
    branch_anchors = [
        extreme_stop_index(coordinates, (-1.0, 0.0)),
        extreme_stop_index(coordinates, (1.0, 0.0)),
        extreme_stop_index(coordinates, (0.0, 1.0)),
        extreme_stop_index(coordinates, (0.0, -1.0)),
    ]

    add_compact_internal_pairs(pairs, center_group, coordinates, 14, 13.0, 6.2)

    branch_groups: List[List[int]] = []
    used = list(center_group)
    for anchor in branch_anchors:
        branch = nearest_stop_indices(coordinates, anchor, 7, excluded=used)
        branch_groups.append(branch)
        used.extend(branch)

    for branch_index, branch in enumerate(branch_groups):
        for stop_index, branch_stop in enumerate(branch):
            center_stop = center_group[(branch_index + stop_index) % len(center_group)]
            value = 14.0 if stop_index < 3 else 6.0
            add_balanced_pair(pairs, center_stop, branch_stop, value)
            if stop_index < 4:
                transfer_center_stop = center_group[(branch_index + stop_index + 2) % len(center_group)]
                add_balanced_pair(pairs, transfer_center_stop, branch_stop, 4.0 if stop_index < 2 else 2.2)
        add_chain_pairs(pairs, branch, value=2.4, skip_value=1.3)
        add_ranked_group_pairs(
            pairs,
            branch,
            branch,
            coordinates,
            pair_count=6,
            values=[3.8] * 3 + [1.8] * 3,
            same_group=True,
        )

    for branch_index in range(len(branch_groups)):
        current_branch = branch_groups[branch_index]
        next_branch = branch_groups[(branch_index + 1) % len(branch_groups)]
        add_ranked_group_pairs(
            pairs,
            current_branch[:3],
            next_branch[:3],
            coordinates,
            pair_count=2,
            values=[2.0, 1.4],
        )

    matrix = matrix_from_balanced_pairs(pairs, base_matrix.shape[0])
    return matrix


def main() -> None:
    base_od_path, road_nodes_path, scenario_dir = get_paths()
    rng = np.random.default_rng(RNG_SEED)

    print("Fitxer OD base utilitzat:")
    print(" -", base_od_path)

    base_df, coordinates = load_reference_data(base_od_path, road_nodes_path)
    base_matrix = base_df.to_numpy(dtype=float)
    total_demand = float(base_matrix.sum())

    print("\nGenerant escenaris OD a:")
    print(" -", scenario_dir)
    print("RNG_SEED =", RNG_SEED)
    print("Reference sparse total demand =", f"{total_demand:.6f}")
    print("Background demand =", BALANCED_BACKGROUND_DEMAND)

    scenarios = {
        "base": generate_balanced_base_matrix(base_matrix, coordinates, total_demand, rng),
        "one_center": generate_balanced_one_center_matrix(base_matrix, coordinates, total_demand, rng),
        "two_centers": generate_balanced_two_centers_matrix(base_matrix, coordinates, total_demand, rng),
        "center_periphery": generate_balanced_center_periphery_matrix(
            base_matrix,
            coordinates,
            total_demand,
            rng,
        ),
    }

    conceptual_summary = {
        "base": "general distributed demand with no dominant center",
        "one_center": "single central demand area with nearby support links",
        "two_centers": "two demand centers with several selected inter-center links",
        "center_periphery": "radial center-periphery demand with local peripheral support",
    }

    written_files = []
    validation_results = []
    for name, matrix in scenarios.items():
        written_path = write_scenario(
            name,
            matrix,
            base_df,
            scenario_dir,
            total_demand,
            off_diagonal_mass(matrix),
        )
        written_files.append(written_path)
        validation_results.append(print_balanced_summary(name, matrix, base_df))

    print("\nResum conceptual dels escenaris:")
    for name in ["base", "one_center", "two_centers", "center_periphery"]:
        print(f" - {name}: {conceptual_summary[name]}")

    print("\nFitxers generats:")
    for path in written_files:
        print(" -", path)

    print("\nValidacio global:", all(validation_results))


if __name__ == "__main__":
    main()

