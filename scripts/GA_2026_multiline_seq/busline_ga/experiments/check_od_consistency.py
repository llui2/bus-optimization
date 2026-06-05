from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

from busline_ga.config.map_cases import resolve_map_case_paths
from busline_ga.config.od_scenarios import get_registered_od_cases, resolve_od_case_path
from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT
from busline_ga.core.network_model import load_network_model


def _load_oneline_style_od(
    bus_stops_path: str,
    od_matrix_path: str,
) -> Tuple[np.ndarray, list[int]]:
    bus_df = pd.read_csv(bus_stops_path)
    od_df = pd.read_csv(od_matrix_path, index_col=0)
    bus_stops = [int(node) for node in bus_df["node"].tolist()]
    od_matrix = od_df.to_numpy(dtype=float)
    return od_matrix, bus_stops


def _print_od_debug(label: str, od_matrix: np.ndarray) -> None:
    print(f"{label} OD shape:", od_matrix.shape)
    print(f"{label} OD total demand:", od_matrix.sum())
    print(f"{label} OD max:", od_matrix.max())
    print(f"{label} OD symmetric:", np.allclose(od_matrix, od_matrix.T))
    print(f"{label} OD diagonal zero:", np.allclose(np.diag(od_matrix), 0))
    print(
        f"{label} OD checksum:",
        np.round(od_matrix.sum(), 6),
        np.round(np.linalg.norm(od_matrix), 6),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", choices=get_registered_od_cases())
    parser.add_argument("--map-case", default="base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    road_nodes_path, road_edges_path = resolve_map_case_paths(
        str(PROJECT_ROOT),
        args.map_case,
        verbose=True,
    )
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), args.od_case)

    oneline_od, oneline_bus_stops = _load_oneline_style_od(
        bus_stops_path=bus_stops_path,
        od_matrix_path=od_matrix_path,
    )
    network = load_network_model(
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        bus_stops_path=bus_stops_path,
        od_matrix_path=od_matrix_path,
    )

    print("od_case:", args.od_case)
    print("map_case:", args.map_case)
    print("od_matrix_path:", od_matrix_path)
    print("bus_stops_order_identical:", oneline_bus_stops == network.bus_stops)
    _print_od_debug("oneline_style", oneline_od)
    _print_od_debug("multiline", network.od_matrix)
    print("OD array_equal:", np.array_equal(oneline_od, network.od_matrix))
    print("OD allclose:", np.allclose(oneline_od, network.od_matrix))


if __name__ == "__main__":
    main()
