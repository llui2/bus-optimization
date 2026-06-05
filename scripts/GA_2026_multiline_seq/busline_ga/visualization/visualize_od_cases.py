from __future__ import annotations

import argparse
import os
import random

import matplotlib.pyplot as plt

from busline_ga.config.map_cases import resolve_map_case_paths
from busline_ga.config.od_scenarios import get_registered_od_cases, resolve_od_case_path
from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, SCRIPT_RESULTS_DIR
from busline_ga.core.network_model import NetworkModel, load_network_model
from busline_ga.visualization.ga_snapshot_plotter import (
    build_edge_paths,
    collect_od_pairs,
    draw_base_network,
    draw_od_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--od-case",
        default="base",
        choices=get_registered_od_cases(),
        help="OD case to preview",
    )
    parser.add_argument("--map-case", default="base", help="Road map case to use")
    parser.add_argument(
        "--od-min-to-plot",
        type=float,
        default=10.0,
        help="Absolute OD demand threshold used only when drawing OD connections.",
    )
    args = parser.parse_args()
    return args


def save_od_preview(
    network: NetworkModel,
    od_case: str,
    output_pdf: str,
    od_min_to_plot: float = 10.0,
) -> None:
    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)
    od_pairs = collect_od_pairs(network, od_min_to_plot=od_min_to_plot)
    fig, ax = plt.subplots(figsize=(9, 9))

    draw_base_network(ax, network, edge_paths)
    draw_od_overlay(
        ax,
        network,
        od_pairs,
        "#B22222",
        od_min_to_plot=od_min_to_plot,
        legend_location="below",
    )
    ax.set_title(f"OD: {od_case}", fontsize=12, pad=8)
    fig.subplots_adjust(bottom=0.10)

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.od_case not in get_registered_od_cases():
        valid_cases = ", ".join(get_registered_od_cases())
        raise ValueError(f"Unknown OD case '{args.od_case}'. Valid cases: {valid_cases}")

    road_nodes_path, road_edges_path = resolve_map_case_paths(
        str(PROJECT_ROOT),
        args.map_case,
        verbose=True,
    )
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), args.od_case)
    network = load_network_model(
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        bus_stops_path=bus_stops_path,
        od_matrix_path=od_matrix_path,
    )

    output_dir = os.path.join(str(SCRIPT_RESULTS_DIR), "od_previews")
    output_pdf = os.path.join(output_dir, f"{args.od_case}.pdf")
    save_od_preview(
        network=network,
        od_case=args.od_case,
        output_pdf=output_pdf,
        od_min_to_plot=args.od_min_to_plot,
    )
    print("OD threshold used for preview:", args.od_min_to_plot)
    print("PDF saved to:", output_pdf)


if __name__ == "__main__":
    main()
