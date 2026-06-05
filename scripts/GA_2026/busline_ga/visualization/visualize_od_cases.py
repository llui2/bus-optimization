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
    parser.add_argument("--od-case", default="base", help="OD case to preview")
    parser.add_argument("--map-case", default="base", help="Road map case to use")
    parser.add_argument(
        "--od-min-to-plot",
        type=float,
        default=10.0,
        help="Minimum OD demand to draw in the preview figure.",
    )
    args = parser.parse_args()
    return args


def save_od_preview(
    network: NetworkModel,
    od_case: str,
    output_pdf: str,
    od_min_to_plot: float = 10.0,
) -> None:
    od_pairs = [
        (stop_i, stop_j, demand)
        for stop_i, stop_j, demand in collect_od_pairs(network)
        if demand >= od_min_to_plot
    ]
    plotted_pair_count = len(od_pairs)

    if plotted_pair_count > 65:
        raise ValueError(
            f"OD case '{od_case}' would plot {plotted_pair_count} OD pairs "
            f"with --od-min-to-plot={od_min_to_plot}. "
            "Increase --od-min-to-plot to avoid a visually saturated preview."
        )

    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)
    fig, ax = plt.subplots(figsize=(9, 9))

    draw_base_network(ax, network, edge_paths)
    draw_od_overlay(ax, network, od_pairs, "#B22222")

    existing_legend = ax.get_legend()
    if existing_legend is not None:
        handles = getattr(existing_legend, "legend_handles", None)
        if handles is None:
            handles = getattr(existing_legend, "legendHandles", [])
        labels = [text.get_text() for text in existing_legend.get_texts()]
        existing_legend.remove()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=3,
                frameon=False,
                fontsize=12,
            )
            fig.subplots_adjust(bottom=0.15)

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
    print("OD minimum demand plotted:", args.od_min_to_plot)
    print(
        "OD pairs plotted:",
        len([
            pair
            for pair in collect_od_pairs(network)
            if pair[2] >= args.od_min_to_plot
        ]),
    )
    print("PDF saved to:", output_pdf)


if __name__ == "__main__":
    main()
