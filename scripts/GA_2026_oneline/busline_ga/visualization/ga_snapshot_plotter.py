from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from busline_ga.config.project_paths import SCRIPT_RESULTS_DIR
from busline_ga.core.network_model import NetworkModel, Node
from busline_ga.core.objective_function import EvaluationResult


Point = Tuple[float, float]
EdgeKey = Tuple[Node, Node]


@dataclass
class ImprovementSnapshotData:
    generation: int
    individual: List[Node]
    fitness: float
    cost: float
    passenger_service: float
    cost_norm: float
    passenger_service_norm: float
    evaluation: EvaluationResult


def jittered_polyline(
    u: Point,
    v: Point,
    rng: random.Random,
    amp: float = 0.08,
    kinks: int = 2,
) -> List[Point]:
    x1, y1 = u
    x2, y2 = v

    points = [(x1, y1)]
    for t in range(1, kinks + 1):
        alpha = t / (kinks + 1)
        xb = x1 + alpha * (x2 - x1)
        yb = y1 + alpha * (y2 - y1)
        dx = x2 - x1
        dy = y2 - y1
        px, py = -dy, dx
        j_perp = rng.uniform(-amp, amp)
        j_free_x = rng.uniform(-amp, amp) * 0.25
        j_free_y = rng.uniform(-amp, amp) * 0.25
        xj = xb + px * j_perp + j_free_x
        yj = yb + py * j_perp + j_free_y
        points.append((xj, yj))

    points.append((x2, y2))
    return points


def normalize_edge(u: Node, v: Node) -> EdgeKey:
    return (u, v) if u <= v else (v, u)


def build_edge_paths(
    network: NetworkModel,
    rng: random.Random,
) -> Dict[EdgeKey, List[Point]]:
    positions = network.positions
    edge_paths: Dict[EdgeKey, List[Point]] = {}

    for u, v in network.graph.edges():
        edge = normalize_edge(u, v)
        if edge not in edge_paths:
            edge_paths[edge] = jittered_polyline(positions[u], positions[v], rng, amp=0.06, kinks=2)

    return edge_paths


def draw_base_network(
    ax,
    network: NetworkModel,
    edge_paths: Dict[EdgeKey, List[Point]],
) -> None:
    positions = network.positions

    for u, v in network.graph.edges():
        edge = normalize_edge(u, v)
        points = edge_paths[edge]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color="0.78", linewidth=1.0, alpha=0.70, zorder=1)

    x_all = [positions[node][0] for node in network.graph.nodes()]
    y_all = [positions[node][1] for node in network.graph.nodes()]
    ax.scatter(
        x_all,
        y_all,
        s=8,
        color="0.86",
        edgecolors="none",
        zorder=1.5,
    )

    xb = [positions[node][0] for node in network.bus_stops]
    yb = [positions[node][1] for node in network.bus_stops]
    ax.scatter(
        xb,
        yb,
        s=38,
        color="#7A0019",
        edgecolors="white",
        linewidths=0.55,
        alpha=0.62,
        zorder=3.0,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.margins(0.03)


def collect_od_pairs(
    network: NetworkModel,
    od_min_to_plot: float = 0.0,
) -> List[Tuple[Node, Node, float]]:
    od_pairs: List[Tuple[Node, Node, float]] = []
    bus_stops = network.bus_stops

    for i in range(len(bus_stops)):
        for j in range(i + 1, len(bus_stops)):
            stop_i = bus_stops[i]
            stop_j = bus_stops[j]
            demand = float(network.get_od_value(stop_i, stop_j))

            if demand >= od_min_to_plot and demand > 0.0:
                od_pairs.append((stop_i, stop_j, demand))

    return od_pairs


def compute_density_center(network: NetworkModel) -> Tuple[float, float]:
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0

    for stop in network.bus_stops:
        stop_weight = 0.0
        for other_stop in network.bus_stops:
            if other_stop == stop:
                continue
            stop_weight += float(network.get_od_value(stop, other_stop))
            stop_weight += float(network.get_od_value(other_stop, stop))

        x, y = network.positions[stop]
        weighted_x += x * stop_weight
        weighted_y += y * stop_weight
        total_weight += stop_weight

    if total_weight > 0.0:
        center = (weighted_x / total_weight, weighted_y / total_weight)
    else:
        x_values = [network.positions[stop][0] for stop in network.bus_stops]
        y_values = [network.positions[stop][1] for stop in network.bus_stops]
        center = (float(np.mean(x_values)), float(np.mean(y_values)))

    return center


def demand_style(
    demand: float,
    low_threshold: float,
    high_threshold: float,
) -> Tuple[float, float]:
    alpha = 0.08
    linewidth = 0.7

    if demand >= high_threshold:
        alpha = 0.35
        linewidth = 3.0
    elif demand >= low_threshold:
        alpha = 0.18
        linewidth = 1.7

    return alpha, linewidth


def draw_density_center(ax, center: Tuple[float, float], color: str) -> None:
    cx, cy = center

    ax.scatter(
        [cx], [cy],
        s=12000,
        color=color,
        alpha=0.05,
        edgecolors="none",
        zorder=2.2,
    )
    ax.scatter(
        [cx], [cy],
        s=4500,
        color=color,
        alpha=0.08,
        edgecolors="none",
        zorder=2.25,
    )
    ax.scatter(
        [cx], [cy],
        s=180,
        color=color,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.8,
        zorder=2.3,
    )


def draw_od_overlay(
    ax,
    network: NetworkModel,
    od_pairs: List[Tuple[Node, Node, float]],
    color: str,
    od_min_to_plot: float = 0.0,
    legend_location: str = "below",
) -> None:
    if not od_pairs:
        return

    demand_values = [demand for _, _, demand in od_pairs]
    low_threshold = float(np.quantile(demand_values, 0.33))
    high_threshold = float(np.quantile(demand_values, 0.66))
    sorted_pairs = sorted(od_pairs, key=lambda item: item[2])

    for stop_i, stop_j, demand in sorted_pairs:
        x1, y1 = network.positions[stop_i]
        x2, y2 = network.positions[stop_j]
        alpha, linewidth = demand_style(demand, low_threshold, high_threshold)
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            alpha=min(alpha, 0.30),
            linewidth=min(linewidth, 2.2),
            zorder=2.6,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=1.4,
            alpha=0.25,
            label=f"Displayed OD pairs: d >= {od_min_to_plot:g}",
        ),
    ]

    if legend_location == "below":
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.035),
            frameon=False,
            ncol=1,
            fontsize=9,
        )
    else:
        ax.legend(handles=legend_elements, loc="upper right", frameon=True)


def get_snapshot_directory(output_dir: Optional[str] = None) -> str:
    snapshot_dir = output_dir

    if snapshot_dir is None:
        snapshot_dir = os.path.join(str(SCRIPT_RESULTS_DIR), "ga_improvements")

    os.makedirs(snapshot_dir, exist_ok=True)
    return snapshot_dir


def get_snapshot_pdf_path(
    generation: int,
    output_dir: Optional[str] = None,
) -> str:
    snapshot_dir = get_snapshot_directory(output_dir)
    pdf_path = os.path.join(snapshot_dir, f"gen_{generation:03d}_best.pdf")
    return pdf_path


def save_improvement_snapshot(
    network: NetworkModel,
    snapshot: ImprovementSnapshotData,
    output_dir: Optional[str] = None,
    od_min_to_plot: float = 10.0,
    show_metrics_title: bool = False,
) -> str:
    positions = network.positions
    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)
    # The OD matrix can include background demand for nearly every stop pair.
    # This threshold filters only the drawing layer so snapshots remain readable.
    od_pairs = collect_od_pairs(network, od_min_to_plot=od_min_to_plot)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_base_network(ax, network, edge_paths)
    draw_od_overlay(ax, network, od_pairs, "#B22222", od_min_to_plot=od_min_to_plot)

    line_edges = [normalize_edge(u, v) for (u, v) in snapshot.evaluation.line_edges]
    for u, v in line_edges:
        points = edge_paths[normalize_edge(u, v)]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color="#7A0019", linewidth=2.0, alpha=0.98, zorder=4.0)

    line_stops = list(snapshot.individual)
    xs_stops = [positions[stop][0] for stop in line_stops]
    ys_stops = [positions[stop][1] for stop in line_stops]
    ax.scatter(
        xs_stops,
        ys_stops,
        s=135,
        color="#FFD23F",
        edgecolors="#7A0019",
        linewidths=1.45,
        zorder=6.5,
    )

    if show_metrics_title:
        title = (
            f"Gen {snapshot.generation:03d} | fit={snapshot.fitness:.12f} | "
            f"cost={snapshot.cost:.12f} | passenger_service={snapshot.passenger_service:.12f}\n"
            f"cost_norm={snapshot.cost_norm:.12f} | "
            f"passenger_service_norm={snapshot.passenger_service_norm:.12f}"
        )
        ax.set_title(title)
    else:
        ax.set_title("")

    out_pdf = get_snapshot_pdf_path(snapshot.generation, output_dir=output_dir)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Snapshot OD threshold used: {od_min_to_plot}")
    print(f"Snapshot OD pairs plotted: {len(od_pairs)}")
    print(f"Snapshot figure saved to: {out_pdf}")
    return out_pdf


def save_final_line_with_od(
    network: NetworkModel,
    individual: List[Node],
    evaluation: EvaluationResult,
    output_path: str,
    title_prefix: str = "LÃ­nia de bus optimitzada",
    od_min_to_plot: float = 10.0,
) -> str:
    positions = network.positions
    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)
    # The OD matrix can include background demand for nearly every stop pair.
    # This threshold filters only the drawing layer so final figures remain readable.
    od_pairs = collect_od_pairs(network, od_min_to_plot=od_min_to_plot)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_base_network(ax, network, edge_paths)
    draw_od_overlay(ax, network, od_pairs, "#B22222", od_min_to_plot=od_min_to_plot)

    line_edges = [normalize_edge(u, v) for (u, v) in evaluation.line_edges]
    for u, v in line_edges:
        points = edge_paths[normalize_edge(u, v)]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color="#7A0019", linewidth=2.0, alpha=0.98, zorder=4.0)

    xs_stops = [positions[stop][0] for stop in individual]
    ys_stops = [positions[stop][1] for stop in individual]
    ax.scatter(
        xs_stops,
        ys_stops,
        s=108,
        color="#FFD23F",
        edgecolors="#7A0019",
        linewidths=1.45,
        zorder=6.5,
    )

    for stop in individual:
        x, y = positions[stop]
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
                "edgecolor": "#7A0019",
                "linewidth": 0.45,
                "alpha": 0.82,
            },
        )

    ax.set_title("")
    fig.subplots_adjust(bottom=0.10)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"OD threshold used for figure: {od_min_to_plot}")
    print(f"OD pairs plotted: {len(od_pairs)}")
    print(f"Final OD figure saved to: {output_path}")
    return output_path

