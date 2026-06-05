from __future__ import annotations

import argparse
import os
import random

import matplotlib.pyplot as plt

from busline_ga.config.map_cases import resolve_map_case_paths
from busline_ga.config.od_scenarios import resolve_od_case_path
from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, SCRIPT_RESULTS_DIR
from busline_ga.core.network_model import NetworkModel, load_network_model
from busline_ga.visualization.ga_snapshot_plotter import build_edge_paths, draw_base_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-case", default="base", help="Road map case to preview")
    args = parser.parse_args()
    return args


def _filesystem_path(path: str) -> str:
    filesystem_path = path

    if os.name == "nt":
        absolute_path = os.path.abspath(path)
        if not absolute_path.startswith("\\\\?\\"):
            filesystem_path = "\\\\?\\" + absolute_path

    return filesystem_path


def _load_network(map_case: str) -> NetworkModel:
    road_nodes_path, road_edges_path = resolve_map_case_paths(
        str(PROJECT_ROOT),
        map_case,
        verbose=True,
    )
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    # The road preview does not draw or use OD demand. The base OD matrix is
    # loaded only because load_network_model expects an OD file.
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), "base")
    network = load_network_model(
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        bus_stops_path=bus_stops_path,
        od_matrix_path=od_matrix_path,
    )
    return network


def save_road_preview(
    network: NetworkModel,
    output_pdf: str,
    output_png: str,
) -> None:
    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_base_network(ax, network, edge_paths)
    os.makedirs(_filesystem_path(os.path.dirname(output_pdf)), exist_ok=True)
    fig.savefig(_filesystem_path(output_pdf), bbox_inches="tight")
    fig.savefig(_filesystem_path(output_png), bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    network = _load_network(args.map_case)
    output_dir = os.path.join(str(SCRIPT_RESULTS_DIR), "road_previews")
    output_pdf = os.path.join(output_dir, f"road_{args.map_case}.pdf")
    output_png = os.path.join(output_dir, f"road_{args.map_case}.png")
    save_road_preview(network, output_pdf, output_png)
    print("PDF saved to:", output_pdf)
    print("PNG saved to:", output_png)


if __name__ == "__main__":
    main()
