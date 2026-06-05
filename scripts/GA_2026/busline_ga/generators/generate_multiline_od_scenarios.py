from __future__ import annotations

import itertools
import math
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from busline_ga.config.project_paths import BUS_NETWORK_DIR, OD_SCENARIOS_DIR, PROJECT_ROOT


Stop = int
Point = Tuple[float, float]
WeightedPair = Tuple[Stop, Stop, float]

SCENARIOS: Dict[str, str] = {
    "multiline_two_centers_sparse": "od_multiline_two_centers_sparse.csv",
    "multiline_three_centers_sparse": "od_multiline_three_centers_sparse.csv",
    "multiline_center_branches_sparse": "od_multiline_center_branches_sparse.csv",
    "multiline_corridors_sparse": "od_multiline_corridors_sparse.csv",
}


def create_empty_od_like(base_od: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=base_od.index.copy(), columns=base_od.columns.copy())
    return matrix


def add_symmetric_demand(
    matrix: pd.DataFrame,
    stop_a: Stop,
    stop_b: Stop,
    value: float,
) -> None:
    if stop_a == stop_b:
        raise ValueError("OD demand cannot be added on the diagonal")

    row = str(stop_a)
    column = str(stop_b)
    if row not in matrix.index or column not in matrix.columns:
        raise KeyError(f"Unknown OD stop pair: {stop_a}, {stop_b}")

    matrix.loc[row, column] = float(matrix.loc[row, column]) + value
    matrix.loc[column, row] = float(matrix.loc[column, row]) + value


def normalize_total_demand(matrix: pd.DataFrame, target_total: float) -> pd.DataFrame:
    current_total = float(matrix.to_numpy(dtype=float).sum())

    if current_total <= 0.0:
        raise ValueError("Cannot normalize an empty OD matrix")

    normalized = matrix.copy()
    normalized *= target_total / current_total
    np.fill_diagonal(normalized.values, 0.0)
    return normalized


def validate_od_matrix(
    matrix: pd.DataFrame,
    target_total: float,
    max_pairs: int = 65,
    base_od: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    values = matrix.to_numpy(dtype=float)
    pair_count = int(np.count_nonzero(np.triu(values, k=1)))
    total = float(values.sum())
    max_value = float(values.max()) if values.size else 0.0
    upper_sum = float(np.triu(values, k=1).sum())
    average_nonzero = upper_sum / pair_count if pair_count > 0 else 0.0
    symmetric = bool(np.allclose(values, values.T))
    diagonal_zero = bool(np.allclose(np.diag(values), 0.0))
    same_labels = True

    if base_od is not None:
        same_labels = (
            list(matrix.index.astype(str)) == list(base_od.index.astype(str))
            and list(matrix.columns.astype(str)) == list(base_od.columns.astype(str))
        )

    if pair_count > max_pairs:
        print(f"[WARNING] OD matrix has {pair_count} undirected pairs; max allowed is {max_pairs}")
        raise ValueError(f"OD matrix has {pair_count} undirected pairs; max allowed is {max_pairs}")
    if not symmetric:
        raise ValueError("OD matrix is not symmetric")
    if not diagonal_zero:
        raise ValueError("OD matrix diagonal is not zero")
    if not same_labels:
        raise ValueError("OD matrix labels do not match od_base.csv")
    if not math.isclose(total, target_total, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"OD total {total:.12f} does not match target {target_total:.12f}")

    result = {
        "total": total,
        "pair_count": pair_count,
        "average_nonzero": average_nonzero,
        "max_value": max_value,
        "symmetric": symmetric,
        "diagonal_zero": diagonal_zero,
        "same_labels": same_labels,
        "passed_validation": True,
    }
    return result


def _load_base_od() -> pd.DataFrame:
    base_path = OD_SCENARIOS_DIR / "od_base.csv"
    base_od = pd.read_csv(base_path, index_col=0)
    base_od.index = base_od.index.astype(str)
    base_od.columns = base_od.columns.astype(str)
    return base_od


def _load_stop_positions() -> Dict[Stop, Point]:
    bus_stops_path = BUS_NETWORK_DIR / "nodes.csv"
    road_nodes_path = PROJECT_ROOT / "data" / "road_network" / "variants" / "base" / "nodes.csv"
    if not road_nodes_path.exists():
        road_nodes_path = PROJECT_ROOT / "data" / "road_network" / "nodes.csv"

    bus_df = pd.read_csv(bus_stops_path)
    nodes_df = pd.read_csv(road_nodes_path)
    stop_set = {int(node) for node in bus_df["node"].tolist()}
    positions = {
        int(row["node"]): (float(row["x"]), float(row["y"]))
        for _, row in nodes_df.iterrows()
        if int(row["node"]) in stop_set
    }
    return positions


def _sorted_stops(
    positions: Dict[Stop, Point],
    score_fn: Callable[[Stop, Point], float],
) -> List[Stop]:
    stops = sorted(positions.keys(), key=lambda stop: (score_fn(stop, positions[stop]), stop))
    return stops


def _nearest_stops(
    positions: Dict[Stop, Point],
    anchor: Point,
    count: int,
    exclude: Iterable[Stop] = (),
) -> List[Stop]:
    excluded = set(exclude)
    ax, ay = anchor
    stops = sorted(
        [stop for stop in positions if stop not in excluded],
        key=lambda stop: ((positions[stop][0] - ax) ** 2 + (positions[stop][1] - ay) ** 2, stop),
    )
    return stops[:count]


def _compact_group(
    positions: Dict[Stop, Point],
    candidates: Sequence[Stop],
    count: int,
    exclude: Iterable[Stop] = (),
) -> List[Stop]:
    excluded = set(exclude)
    available = [stop for stop in candidates if stop not in excluded]
    cx = float(np.mean([positions[stop][0] for stop in available]))
    cy = float(np.mean([positions[stop][1] for stop in available]))
    group = _nearest_stops(positions, (cx, cy), count, exclude=excluded)
    return group


def _add_pairs(matrix: pd.DataFrame, pairs: Sequence[WeightedPair]) -> None:
    seen = set()

    for stop_a, stop_b, value in pairs:
        key = tuple(sorted((stop_a, stop_b)))
        if key not in seen:
            add_symmetric_demand(matrix, stop_a, stop_b, value)
            seen.add(key)


def _chain_pairs(stops: Sequence[Stop], weight: float) -> List[WeightedPair]:
    pairs = [(stops[index], stops[index + 1], weight) for index in range(len(stops) - 1)]
    return pairs


def _skip_pairs(stops: Sequence[Stop], step: int, weight: float) -> List[WeightedPair]:
    pairs = [(stops[index], stops[index + step], weight) for index in range(len(stops) - step)]
    return pairs


def _sparse_internal_pairs(
    positions: Dict[Stop, Point],
    stops: Sequence[Stop],
    target_count: int,
    rng: random.Random,
    strong_range: Tuple[float, float],
) -> List[WeightedPair]:
    candidates = []

    for stop_a, stop_b in itertools.combinations(stops, 2):
        ax, ay = positions[stop_a]
        bx, by = positions[stop_b]
        distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        candidates.append((distance, stop_a, stop_b))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = candidates[:target_count]
    pairs = [
        (stop_a, stop_b, rng.uniform(strong_range[0], strong_range[1]))
        for _, stop_a, stop_b in selected
    ]
    return pairs


def _scenario_two_centers(base_od: pd.DataFrame, positions: Dict[Stop, Point], rng: random.Random) -> pd.DataFrame:
    matrix = create_empty_od_like(base_od)
    left_candidates = _sorted_stops(positions, lambda _, point: point[0])[:24]
    right_candidates = _sorted_stops(positions, lambda _, point: -point[0])[:24]
    left_center = _compact_group(positions, left_candidates, 9)
    right_center = _compact_group(positions, right_candidates, 9, exclude=left_center)

    pairs = []
    pairs.extend(_sparse_internal_pairs(positions, left_center, 18, rng, (5.2, 8.0)))
    pairs.extend(_sparse_internal_pairs(positions, right_center, 18, rng, (5.2, 8.0)))
    pairs.extend([
        (left_center[1], right_center[1], 3.6),
        (left_center[2], right_center[2], 3.3),
        (left_center[3], right_center[3], 3.0),
        (left_center[4], right_center[4], 2.7),
        (left_center[5], right_center[5], 2.4),
        (left_center[6], right_center[6], 2.0),
        (left_center[0], right_center[7], 1.7),
        (left_center[7], right_center[0], 1.7),
    ])
    _add_pairs(matrix, pairs)
    return matrix


def _scenario_three_centers(base_od: pd.DataFrame, positions: Dict[Stop, Point], rng: random.Random) -> pd.DataFrame:
    matrix = create_empty_od_like(base_od)
    left_bottom = _nearest_stops(positions, (3.0, 4.0), 8)
    top_middle = _nearest_stops(positions, (10.0, 16.5), 8, exclude=left_bottom)
    right_bottom = _nearest_stops(positions, (16.0, 5.0), 8, exclude=[*left_bottom, *top_middle])
    transfer_ab = _nearest_stops(positions, (6.5, 10.0), 2, exclude=[*left_bottom, *top_middle, *right_bottom])
    transfer_bc = _nearest_stops(
        positions,
        (13.0, 10.5),
        2,
        exclude=[*left_bottom, *top_middle, *right_bottom, *transfer_ab],
    )
    transfer_ac = _nearest_stops(
        positions,
        (9.5, 5.0),
        1,
        exclude=[*left_bottom, *top_middle, *right_bottom, *transfer_ab, *transfer_bc],
    )

    pairs = []
    pairs.extend(_sparse_internal_pairs(positions, left_bottom, 13, rng, (5.0, 7.6)))
    pairs.extend(_sparse_internal_pairs(positions, top_middle, 13, rng, (5.0, 7.6)))
    pairs.extend(_sparse_internal_pairs(positions, right_bottom, 13, rng, (5.0, 7.6)))
    pairs.extend([
        (left_bottom[2], top_middle[2], 4.4),
        (left_bottom[3], top_middle[3], 4.0),
        (left_bottom[4], top_middle[4], 3.6),
        (left_bottom[1], top_middle[5], 3.0),
        (top_middle[2], right_bottom[2], 4.4),
        (top_middle[3], right_bottom[3], 4.0),
        (top_middle[4], right_bottom[4], 3.6),
        (top_middle[1], right_bottom[5], 3.0),
        (left_bottom[5], right_bottom[5], 2.6),
        (left_bottom[6], right_bottom[6], 2.3),
        (left_bottom[1], right_bottom[4], 2.0),
        (left_bottom[2], transfer_ab[0], 2.1),
        (transfer_ab[0], top_middle[1], 2.1),
        (left_bottom[4], transfer_ab[1], 1.7),
        (transfer_ab[1], top_middle[4], 1.7),
        (top_middle[5], transfer_bc[0], 2.1),
        (transfer_bc[0], right_bottom[2], 2.1),
        (top_middle[6], transfer_bc[1], 1.7),
        (transfer_bc[1], right_bottom[4], 1.7),
        (left_bottom[6], transfer_ac[0], 1.5),
        (transfer_ac[0], right_bottom[6], 1.5),
    ])
    _add_pairs(matrix, pairs)
    return matrix


def _scenario_center_branches(base_od: pd.DataFrame, positions: Dict[Stop, Point], rng: random.Random) -> pd.DataFrame:
    matrix = create_empty_od_like(base_od)
    center = _nearest_stops(positions, (10.0, 10.0), 7)
    north_branch = _nearest_stops(positions, (10.0, 18.0), 6, exclude=center)
    west_branch = _nearest_stops(positions, (2.0, 8.0), 6, exclude=[*center, *north_branch])
    east_branch = _nearest_stops(positions, (18.0, 8.0), 6, exclude=[*center, *north_branch, *west_branch])

    pairs = []
    ordered_branches = []
    for branch in [north_branch, west_branch, east_branch]:
        ordered_branch = sorted(
            branch,
            key=lambda stop: ((positions[stop][0] - 10.0) ** 2 + (positions[stop][1] - 10.0) ** 2, stop),
        )
        ordered_branches.append(ordered_branch)
        for index, branch_stop in enumerate(ordered_branch):
            center_stop = center[index % len(center)]
            pairs.append((branch_stop, center_stop, rng.uniform(5.0, 7.8)))
        pairs.append((ordered_branch[0], center[(len(ordered_branch) + 1) % len(center)], 4.6))
        pairs.append((ordered_branch[2], center[(len(ordered_branch) + 2) % len(center)], 4.1))
        pairs.append((ordered_branch[-1], center[(len(ordered_branch) + 4) % len(center)], 3.8))
        pairs.extend(_chain_pairs(ordered_branch, 1.7))
        pairs.extend(_skip_pairs(ordered_branch, 2, 1.1)[:2])

    pairs.extend(_chain_pairs(center, 2.2))
    pairs.extend([
        (ordered_branches[0][1], ordered_branches[1][1], 1.2),
        (ordered_branches[0][2], ordered_branches[2][1], 1.2),
        (ordered_branches[1][2], ordered_branches[2][2], 1.0),
        (ordered_branches[1][-1], ordered_branches[2][-1], 0.9),
    ])
    _add_pairs(matrix, pairs)
    return matrix


def _corridor_stops(
    positions: Dict[Stop, Point],
    score_fn: Callable[[Stop, Point], float],
    order_fn: Callable[[Stop, Point], float],
    count: int,
    exclude: Iterable[Stop] = (),
) -> List[Stop]:
    excluded = set(exclude)
    candidates = [
        stop
        for stop in _sorted_stops(positions, score_fn)
        if stop not in excluded
    ][:count]
    stops = sorted(candidates, key=lambda stop: (order_fn(stop, positions[stop]), stop))
    return stops


def _scenario_corridors(base_od: pd.DataFrame, positions: Dict[Stop, Point], rng: random.Random) -> pd.DataFrame:
    matrix = create_empty_od_like(base_od)
    bottom = _corridor_stops(
        positions,
        score_fn=lambda _, point: point[1],
        order_fn=lambda _, point: point[0],
        count=10,
    )
    middle = _corridor_stops(
        positions,
        score_fn=lambda _, point: abs(point[1] - 10.0),
        order_fn=lambda _, point: point[0],
        count=10,
        exclude=bottom,
    )
    top = _corridor_stops(
        positions,
        score_fn=lambda _, point: -point[1],
        order_fn=lambda _, point: point[0],
        count=10,
        exclude=[*bottom, *middle],
    )

    pairs = []
    for corridor in [bottom, middle, top]:
        pairs.extend((stop_a, stop_b, rng.uniform(4.2, 6.5)) for stop_a, stop_b, _ in _chain_pairs(corridor, 1.0))
        pairs.extend((stop_a, stop_b, rng.uniform(2.6, 4.0)) for stop_a, stop_b, _ in _skip_pairs(corridor, 2, 1.0)[:6])
        pairs.extend((stop_a, stop_b, rng.uniform(1.7, 2.6)) for stop_a, stop_b, _ in _skip_pairs(corridor, 3, 1.0)[:3])

    pairs.extend([
        (bottom[len(bottom) // 2], middle[len(middle) // 2], 1.8),
        (middle[len(middle) // 2], top[len(top) // 2], 1.8),
        (bottom[len(bottom) // 2 - 1], middle[len(middle) // 2 - 1], 1.2),
        (middle[len(middle) // 2 + 1], top[len(top) // 2 + 1], 1.2),
    ])
    _add_pairs(matrix, pairs)
    return matrix


def _write_scenario(
    name: str,
    filename: str,
    raw_matrix: pd.DataFrame,
    target_total: float,
    base_od: pd.DataFrame,
) -> None:
    normalized = normalize_total_demand(raw_matrix, target_total)
    summary = validate_od_matrix(normalized, target_total, max_pairs=65, base_od=base_od)
    output_path = OD_SCENARIOS_DIR / filename
    normalized.to_csv(output_path)
    print(
        f"{name}: filename={filename} "
        f"total_demand={summary['total']:.6f} "
        f"od_pairs={summary['pair_count']} "
        f"average_nonzero={summary['average_nonzero']:.6f} "
        f"max_od={summary['max_value']:.6f} "
        f"diagonal_zero={summary['diagonal_zero']} "
        f"symmetric={summary['symmetric']} "
        f"passed_validation={summary['passed_validation']}"
    )


def main() -> None:
    rng = random.Random(20260518)
    base_od = _load_base_od()
    positions = _load_stop_positions()
    target_total = float(base_od.to_numpy(dtype=float).sum())
    OD_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    builders = {
        "multiline_two_centers_sparse": _scenario_two_centers,
        "multiline_three_centers_sparse": _scenario_three_centers,
        "multiline_center_branches_sparse": _scenario_center_branches,
        "multiline_corridors_sparse": _scenario_corridors,
    }

    for name, filename in SCENARIOS.items():
        raw_matrix = builders[name](base_od, positions, rng)
        _write_scenario(name, filename, raw_matrix, target_total, base_od)


if __name__ == "__main__":
    main()
