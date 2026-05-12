from __future__ import annotations

import argparse
import os
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from busline_ga.core.line_builder import build_line_from_stops
from busline_ga.core.network_model import NetworkModel, Node
from busline_ga.core.objective_function import EvaluationResult
from busline_ga.experiments.main_multiline_ga import run_multiline_ga_experiment
from busline_ga.visualization.ga_snapshot_plotter import (
    build_edge_paths,
    collect_od_pairs,
    compute_density_center,
    draw_base_network,
    draw_density_center,
    draw_od_overlay,
    normalize_edge,
)


LINE_COLORS = [
    "#0F766E",
    "#C2410C",
    "#7C3AED",
    "#BE123C",
    "#1D4ED8",
    "#4D7C0F",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", help="Cas OD a executar")
    parser.add_argument("--map-case", default="base", help="Variant de xarxa viaria")
    parser.add_argument(
        "--n-lines",
        "--lines",
        dest="n_lines",
        type=int,
        default=2,
        help="Nombre de linies a optimitzar",
    )
    parser.add_argument("--line-length", type=int, default=6, help="Parades per linia")
    parser.add_argument("--population-size", type=int, default=150, help="Mida de poblacio")
    parser.add_argument("--generations", type=int, default=100, help="Nombre de generacions")
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.5,
        help="Pes de cost a la funcio objectiu",
    )
    parser.add_argument("--seed", type=int, default=42, help="Llavor del GA")
    parser.add_argument(
        "--shared-stop-penalty",
        type=float,
        default=0.0,
        help="Optional fitness penalty for reusing stops from previous fixed lines.",
    )
    args = parser.parse_args()

    if args.n_lines < 1:
        parser.error("--n-lines / --lines must be >= 1")
    if args.line_length < 2:
        parser.error("--line-length must be >= 2")

    return args


def draw_line_edges(
    ax,
    network: NetworkModel,
    edge_paths,
    line_edges: Sequence[Tuple[Node, Node]],
    color: str,
    linewidth: float,
    zorder: float,
) -> None:
    normalized_edges = [normalize_edge(u, v) for u, v in line_edges]

    for edge in normalized_edges:
        points = edge_paths[edge]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=linewidth,
            alpha=0.96,
            zorder=zorder,
        )


def draw_line_stops(
    ax,
    network: NetworkModel,
    line: Sequence[Node],
    color: str,
    label: str,
) -> None:
    xs = [network.positions[stop][0] for stop in line]
    ys = [network.positions[stop][1] for stop in line]
    ax.scatter(
        xs,
        ys,
        s=118,
        color=color,
        edgecolors="white",
        linewidths=1.4,
        zorder=6.4,
        label=label,
    )


def draw_stop_labels(
    ax,
    network: NetworkModel,
    line: Sequence[Node],
    color: str,
) -> None:
    for stop in line:
        x, y = network.positions[stop]
        ax.text(
            x + 0.10,
            y + 0.10,
            str(stop),
            fontsize=8,
            fontweight="bold",
            color="black",
            ha="left",
            va="bottom",
            zorder=7.0,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.55,
                "alpha": 0.84,
            },
        )


def format_line_label(index: int, evaluation: EvaluationResult) -> str:
    label = (
        f"Line {index} | fit={evaluation.fitness:.3f} | "
        f"cost={evaluation.cost:.2f} | service={evaluation.passenger_service:.2f}"
    )
    return label


def save_multiline_map(
    network: NetworkModel,
    lines: Sequence[Sequence[Node]],
    evaluations: Sequence[EvaluationResult],
    output_pdf: str,
    output_png: str,
    title: str,
) -> None:
    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)
    od_pairs = collect_od_pairs(network)
    density_center = compute_density_center(network)
    fig, ax = plt.subplots(figsize=(9, 9))

    draw_base_network(ax, network, edge_paths)
    draw_density_center(ax, density_center, "#B22222")
    draw_od_overlay(ax, network, od_pairs, "#B22222")

    line_handles: List[Line2D] = []

    for index, line in enumerate(lines, start=1):
        evaluation = evaluations[index - 1]
        color = LINE_COLORS[(index - 1) % len(LINE_COLORS)]
        line_data = build_line_from_stops(network, list(line))
        linewidth = 2.1 + 0.30 * (index - 1)
        zorder = 4.2 + 0.2 * index
        draw_line_edges(
            ax=ax,
            network=network,
            edge_paths=edge_paths,
            line_edges=line_data["line_edges"],
            color=color,
            linewidth=linewidth,
            zorder=zorder,
        )
        draw_line_stops(
            ax=ax,
            network=network,
            line=line,
            color=color,
            label=format_line_label(index, evaluation),
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
                label=format_line_label(index, evaluation),
            )
        )

    existing_legend = ax.get_legend()
    if existing_legend is not None:
        ax.add_artist(existing_legend)

    ax.legend(handles=line_handles, loc="upper left", frameon=True, fontsize=8)
    ax.set_title(title)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_result = run_multiline_ga_experiment(
        od_case=args.od_case,
        map_case=args.map_case,
        n_lines=args.n_lines,
        line_length=args.line_length,
        population_size=args.population_size,
        generations=args.generations,
        lambda_=args.lambda_,
        seed=args.seed,
        shared_stop_penalty=args.shared_stop_penalty,
    )
    results_dir = run_result["results_dir"]
    output_pdf = os.path.join(results_dir, "multiline_map.pdf")
    output_png = os.path.join(results_dir, "multiline_map.png")
    title = (
        f"Multi-line GA | od={args.od_case} | map={args.map_case} | "
        f"n_lines={args.n_lines} | lambda={args.lambda_} | "
        f"shared_stop_penalty={args.shared_stop_penalty} | "
        f"adjusted_service={run_result['system_metrics']['adjusted_passenger_service']:.2f} | "
        f"total_cost={run_result['system_metrics']['total_route_cost']:.2f} | "
        f"fitness={run_result['system_metrics']['system_fitness']:.3f}"
    )
    save_multiline_map(
        network=run_result["network"],
        lines=run_result["lines"],
        evaluations=run_result["evaluations"],
        output_pdf=output_pdf,
        output_png=output_png,
        title=title,
    )
    print("PDF guardat a:", output_pdf)
    print("PNG guardat a:", output_png)


if __name__ == "__main__":
    main()
