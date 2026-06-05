from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from busline_ga.core.line_builder import build_line_from_stops
from busline_ga.core.network_model import NetworkModel, Node
from busline_ga.visualization.ga_snapshot_plotter import (
    build_edge_paths,
    draw_base_network,
    draw_od_overlay,
)
from busline_ga.visualization.visualize_multiline_ga import (
    draw_line_edges,
    draw_line_stops,
    draw_stop_labels,
)


LINE_COLORS = [
    "#0F766E",
    "#C2410C",
    "#7C3AED",
    "#BE123C",
    "#1D4ED8",
    "#4D7C0F",
]


def _filesystem_path(path: str) -> str:
    filesystem_path = path

    if os.name == "nt":
        absolute_path = os.path.abspath(path)
        if not absolute_path.startswith("\\\\?\\"):
            filesystem_path = "\\\\?\\" + absolute_path

    return filesystem_path


def _iter_od_pairs_to_plot(
    network: NetworkModel,
    od_threshold: float,
) -> List[Tuple[Node, Node, float]]:
    od_pairs: List[Tuple[Node, Node, float]] = []
    bus_stops = list(network.bus_stops)

    for i, stop_i in enumerate(bus_stops):
        for stop_j in bus_stops[i + 1:]:
            demand = float(network.get_od_value(stop_i, stop_j))
            if demand >= od_threshold:
                od_pairs.append((stop_i, stop_j, demand))

    return od_pairs


def _draw_filtered_od_overlay(
    ax,
    network: NetworkModel,
    od_threshold: float,
) -> int:
    od_pairs = _iter_od_pairs_to_plot(network, od_threshold)
    draw_od_overlay(ax, network, od_pairs, "#B22222")
    return len(od_pairs)


def save_joint_multiline_map(
    network: NetworkModel,
    lines: Sequence[Sequence[Node]],
    output_pdf: str,
    output_png: str,
    show_od: bool = True,
    od_threshold: float = 3.0,
) -> None:
    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_base_network(ax, network, edge_paths)
    line_handles: List[Line2D] = []
    od_pair_count = 0

    if show_od:
        od_pair_count = _draw_filtered_od_overlay(
            ax=ax,
            network=network,
            od_threshold=od_threshold,
        )

    for index, line in enumerate(lines, start=1):
        color = LINE_COLORS[(index - 1) % len(LINE_COLORS)]
        line_data = build_line_from_stops(network, list(line))
        linewidth = 2.1 + 0.30 * (index - 1)
        line_label = f"Línia {index}"
        draw_line_edges(
            ax=ax,
            network=network,
            edge_paths=edge_paths,
            line_edges=line_data["line_edges"],
            color=color,
            linewidth=linewidth,
            zorder=4.2 + 0.2 * index,
        )
        draw_line_stops(
            ax=ax,
            network=network,
            line=line,
            color=color,
            label=line_label,
        )
        draw_stop_labels(ax=ax, network=network, line=line, color=color)
        line_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=linewidth,
                marker="o",
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="white",
                label=line_label,
            )
        )

    ncol = min(4, max(1, len(line_handles)))
    ax.legend(
        handles=line_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=ncol,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(_filesystem_path(output_pdf), bbox_inches="tight")
    fig.savefig(_filesystem_path(output_png), bbox_inches="tight", dpi=220)
    plt.close(fig)

    if show_od:
        print(
            "Joint map OD overlay:",
            f"threshold={od_threshold}",
            f"plotted_pairs={od_pair_count}",
            f"pdf={output_pdf}",
            f"png={output_png}",
        )


def save_joint_multiline_map_from_run_result(
    run_result: Dict[str, Any],
    output_pdf: Optional[str] = None,
    output_png: Optional[str] = None,
    show_od: bool = True,
    od_threshold: float = 3.0,
) -> Tuple[str, str]:
    results_dir = run_result["results_dir"]
    pdf_path = output_pdf or os.path.join(results_dir, "joint_multiline_map.pdf")
    png_path = output_png or os.path.join(results_dir, "joint_multiline_map.png")
    save_joint_multiline_map(
        network=run_result["network"],
        lines=run_result["lines"],
        output_pdf=pdf_path,
        output_png=png_path,
        show_od=show_od,
        od_threshold=od_threshold,
    )
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base")
    parser.add_argument("--map-case", default="base")
    parser.add_argument("--n-lines", "--lines", dest="n_lines", type=int, default=2)
    parser.add_argument("--line-length", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=150)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mutation-prob", type=float, default=0.20)
    parser.add_argument("--crossover-prob", type=float, default=0.80)
    parser.add_argument("--compactness-penalty", type=float, default=0.0)
    parser.add_argument("--max-segment-cost-soft", type=float, default=None)
    parser.add_argument("--max-line-cost-soft", type=float, default=None)
    parser.add_argument("--max-edges-per-line-soft", type=float, default=None)
    parser.add_argument("--n-elite", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-improvement-snapshots", action="store_true")
    parser.add_argument("--save-evolution-plots", action="store_true")
    parser.add_argument("--od-min-demand", type=float, default=3.0)
    args = parser.parse_args()

    if args.n_lines < 1:
        parser.error("--n-lines / --lines must be >= 1")
    if args.line_length < 2:
        parser.error("--line-length must be >= 2")

    return args


def main() -> None:
    args = parse_args()
    from busline_ga.experiments.main_joint_multiline_ga import run_joint_multiline_ga_experiment

    run_result = run_joint_multiline_ga_experiment(
        od_case=args.od_case,
        map_case=args.map_case,
        n_lines=args.n_lines,
        line_length=args.line_length,
        population_size=args.population_size,
        generations=args.generations,
        lambda_=args.lambda_,
        seed=args.seed,
        mutation_prob=args.mutation_prob,
        crossover_prob=args.crossover_prob,
        compactness_penalty=args.compactness_penalty,
        max_segment_cost_soft=args.max_segment_cost_soft,
        max_line_cost_soft=args.max_line_cost_soft,
        max_edges_per_line_soft=args.max_edges_per_line_soft,
        n_elite=args.n_elite,
        tournament_size=args.tournament_size,
        run_name=args.run_name,
        save_improvement_snapshots=args.save_improvement_snapshots,
        save_evolution_plots=args.save_evolution_plots,
    )
    pdf_path, png_path = save_joint_multiline_map_from_run_result(
        run_result,
        od_threshold=args.od_min_demand,
    )
    print("PDF saved to:", pdf_path)
    print("PNG saved to:", png_path)


if __name__ == "__main__":
    main()
