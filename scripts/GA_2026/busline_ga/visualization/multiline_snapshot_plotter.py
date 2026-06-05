from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from busline_ga.core.line_builder import build_line_from_stops
from busline_ga.core.network_model import NetworkModel, Node
from busline_ga.core.objective_function import EvaluationResult
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
CANDIDATE_COLOR = "#FFD23F"


def _filesystem_path(path: str) -> str:
    filesystem_path = path

    if os.name == "nt":
        absolute_path = os.path.abspath(path)
        if not absolute_path.startswith("\\\\?\\"):
            filesystem_path = "\\\\?\\" + absolute_path

    return filesystem_path


def _draw_line_edges(
    ax,
    edge_paths,
    line_edges,
    color: str,
    linewidth: float,
    zorder: float,
    alpha: float = 0.96,
) -> None:
    for u, v in line_edges:
        edge = normalize_edge(u, v)
        points = edge_paths[edge]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _draw_line_stops(
    ax,
    network: NetworkModel,
    line: Sequence[Node],
    color: str,
    edge_color: str,
    size: float,
    zorder: float,
) -> None:
    xs = [network.positions[stop][0] for stop in line]
    ys = [network.positions[stop][1] for stop in line]
    ax.scatter(
        xs,
        ys,
        s=size,
        color=color,
        edgecolors=edge_color,
        linewidths=1.25,
        zorder=zorder,
    )


def _draw_stop_labels(
    ax,
    network: NetworkModel,
    lines: Sequence[Sequence[Node]],
) -> None:
    labeled_stops = []

    for line in lines:
        for stop in line:
            if stop not in labeled_stops:
                labeled_stops.append(stop)

    for stop in labeled_stops:
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
                "edgecolor": "#333333",
                "linewidth": 0.45,
                "alpha": 0.84,
            },
        )


def _shared_stop_penalty(
    fixed_lines: Sequence[Sequence[Node]],
    candidate_line: Sequence[Node],
    penalty_weight: float,
) -> float:
    fixed_stops = {stop for line in fixed_lines for stop in line}
    candidate_stops = set(candidate_line)
    shared_count = sum(1 for stop in candidate_stops if stop in fixed_stops)
    ratio = shared_count / max(1, len(candidate_stops))
    penalty = penalty_weight * ratio
    return penalty


def _format_title(
    line_index: int,
    generation: int,
    candidate_evaluation: EvaluationResult,
    title_context: Optional[Dict[str, Any]],
    shared_stop_penalty: float,
) -> str:
    context = title_context or {}
    od_case = context.get("od_case", "")
    map_case = context.get("map_case", "")
    title = (
        f"Multi-line GA improvement | od_case={od_case} | map_case={map_case} | "
        f"line={line_index:02d} | generation={generation:03d}\n"
        f"fitness={candidate_evaluation.fitness:.12f} | "
        f"cost={candidate_evaluation.cost:.12f} | "
        f"passenger_service={candidate_evaluation.passenger_service:.12f}\n"
        f"cost_norm={candidate_evaluation.cost_norm:.12f} | "
        f"service_norm={candidate_evaluation.service_norm:.12f} | "
        f"shared_stop_penalty={shared_stop_penalty:.12f}"
    )
    return title


def save_multiline_improvement_snapshot(
    network: NetworkModel,
    fixed_lines: Sequence[Sequence[Node]],
    candidate_line: Sequence[Node],
    fixed_evaluations: Sequence[EvaluationResult],
    candidate_evaluation: EvaluationResult,
    output_path: str,
    line_index: int,
    generation: int,
    title_context: Optional[Dict[str, Any]] = None,
) -> str:
    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)
    od_pairs = collect_od_pairs(network)
    density_center = compute_density_center(network)
    context = title_context or {}
    penalty_weight = float(context.get("shared_stop_penalty", 0.0))
    shared_penalty = _shared_stop_penalty(fixed_lines, candidate_line, penalty_weight)

    os.makedirs(_filesystem_path(os.path.dirname(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_base_network(ax, network, edge_paths)
    draw_density_center(ax, density_center, "#B22222")
    draw_od_overlay(ax, network, od_pairs, "#B22222")

    legend_handles: List[Line2D] = []
    for fixed_index, fixed_line in enumerate(fixed_lines, start=1):
        color = LINE_COLORS[(fixed_index - 1) % len(LINE_COLORS)]
        if fixed_index - 1 < len(fixed_evaluations):
            fixed_edges = fixed_evaluations[fixed_index - 1].line_edges
        else:
            fixed_edges = build_line_from_stops(network, list(fixed_line))["line_edges"]
        _draw_line_edges(ax, edge_paths, fixed_edges, color=color, linewidth=2.0, zorder=4.0)
        _draw_line_stops(ax, network, fixed_line, color=color, edge_color="white", size=105, zorder=6.0)
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.0, marker="o", markersize=7, label=f"Fixed line {fixed_index:02d}")
        )

    _draw_line_edges(
        ax,
        edge_paths,
        candidate_evaluation.line_edges,
        color="#7A0019",
        linewidth=2.8,
        zorder=5.2,
        alpha=0.98,
    )
    _draw_line_stops(
        ax,
        network,
        candidate_line,
        color=CANDIDATE_COLOR,
        edge_color="#7A0019",
        size=145,
        zorder=6.8,
    )
    _draw_stop_labels(ax, network, [*fixed_lines, candidate_line])
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#7A0019",
            lw=2.8,
            marker="o",
            markersize=8,
            markerfacecolor=CANDIDATE_COLOR,
            markeredgecolor="#7A0019",
            label=f"Improved candidate line {line_index:02d}",
        )
    )

    existing_legend = ax.get_legend()
    if existing_legend is not None:
        ax.add_artist(existing_legend)
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=8)
    ax.set_title(_format_title(line_index, generation, candidate_evaluation, context, shared_penalty))

    fig.savefig(_filesystem_path(output_path), bbox_inches="tight")
    plt.close(fig)
    return output_path
