from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from busline_ga.config.project_paths import BUS_NETWORK_DIR, PROJECT_ROOT, SCRIPT_DIR
from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.line_builder import build_line_from_stops
from busline_ga.config.map_cases import get_road_experiment_results_dir, resolve_map_case_paths
from busline_ga.core.network_model import Node, load_network_model
from busline_ga.core.objective_function import ObjectiveFunction
from busline_ga.config.od_scenarios import resolve_od_case_path


FamilyRecord = Dict[str, object]
Point = Tuple[float, float]
EdgeKey = Tuple[Node, Node]


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


def get_paths(od_case: str, map_case: str) -> Tuple[str, str, str, str, str]:
    case_results_dir = get_road_experiment_results_dir(str(SCRIPT_DIR), od_case, map_case)
    results_dir = os.path.join(case_results_dir, "initial_population")
    os.makedirs(results_dir, exist_ok=True)

    road_nodes_path, road_edges_path = resolve_map_case_paths(str(PROJECT_ROOT), map_case, verbose=True)
    bus_stops_path = os.path.join(str(BUS_NETWORK_DIR), "nodes.csv")
    od_matrix_path = resolve_od_case_path(str(PROJECT_ROOT), od_case)

    return road_nodes_path, road_edges_path, bus_stops_path, od_matrix_path, results_dir


def build_edge_paths(
    network,
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
    network,
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


def save_figure(fig, output_base: str) -> str:
    out_pdf = f"{output_base}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def build_family_sample(records: Sequence[FamilyRecord], per_family: int) -> List[FamilyRecord]:
    grouped: Dict[str, List[FamilyRecord]] = {}
    sample_records: List[FamilyRecord] = []

    for record in records:
        family = str(record["family"])
        if family not in grouped:
            grouped[family] = []
        grouped[family].append(record)

    for family in ["service_best", "service", "demand", "spatial", "hybrid", "random"]:
        family_records = grouped.get(family, [])
        sample_records.extend(family_records[:per_family])

    return sample_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-case", default="base", help="Cas OD a visualitzar")
    parser.add_argument("--map-case", default="base", help="Variant de xarxa viÃ ria")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        road_nodes_path,
        road_edges_path,
        bus_stops_path,
        od_matrix_path,
        results_dir,
    ) = get_paths(args.od_case, args.map_case)

    print("Carregant dades...")
    print("Cas OD =", args.od_case)
    print("Cas mapa =", args.map_case)
    network = load_network_model(
        road_nodes_path,
        road_edges_path,
        bus_stops_path,
        od_matrix_path,
    )

    print("Preparant funcio objectiu...")
    objective = ObjectiveFunction(network)
    objective.precompute_edge_costs()
    objective.precompute_edge_service()

    ga = GeneticOptimizer(
        network=network,
        objective_function=objective,
        line_length=6,
        population_size=50,
        seed=42,
        init_ratio_service=0.30,
        init_ratio_demand=0.25,
        init_ratio_spatial=0.15,
        init_ratio_hybrid=0.15,
        init_ratio_random=0.15,
        inject_best_service_candidate=True,
        nearest_neighbors_k=8,
        init_top_k=8,
    )

    print("Generant poblacio inicial...")
    ga.generate_initial_population()

    family_colors = {
        "service_best": "#4C1D95",
        "service": "#8B5CF6",
        "demand": "#1B6CA8",
        "spatial": "#2E8B57",
        "hybrid": "#D97706",
        "random": "#6B7280",
    }

    positions = network.positions
    rng = random.Random(42)
    edge_paths = build_edge_paths(network, rng)

    print("Creant Figura A...")
    fig_a, ax_a = plt.subplots(figsize=(8, 8))
    draw_base_network(ax_a, network, edge_paths)

    for family in ["service_best", "service", "demand", "spatial", "hybrid", "random"]:
        family_records = [
            record for record in ga.population_records if str(record["family"]) == family
        ]
        seed_stops = [int(record["seed_stop"]) for record in family_records if record["seed_stop"] is not None]

        if not seed_stops:
            continue

        xs = [positions[stop][0] for stop in seed_stops]
        ys = [positions[stop][1] for stop in seed_stops]
        ax_a.scatter(
            xs,
            ys,
            s=92,
            color=family_colors[family],
            edgecolors="white",
            linewidths=0.9,
            alpha=0.90,
            zorder=5.5,
            label=family.capitalize(),
        )

    ax_a.legend(frameon=True, loc="upper right")
    ax_a.set_title("Parades llavor de la poblacio inicial")
    seed_pdf = save_figure(
        fig_a,
        os.path.join(results_dir, "initial_population_seed_stops"),
    )

    print("Creant Figura B...")
    stop_frequency = Counter()
    for record in ga.population_records:
        individual = list(record["individual"])
        stop_frequency.update(individual)

    fig_b, ax_b = plt.subplots(figsize=(8, 8))
    draw_base_network(ax_b, network, edge_paths)

    frequent_stops = [stop for stop in network.bus_stops if stop_frequency.get(stop, 0) > 0]
    xs_freq = [positions[stop][0] for stop in frequent_stops]
    ys_freq = [positions[stop][1] for stop in frequent_stops]
    counts = [stop_frequency[stop] for stop in frequent_stops]
    sizes = [45 + 18 * count for count in counts]

    scatter = ax_b.scatter(
        xs_freq,
        ys_freq,
        s=sizes,
        c=counts,
        cmap="YlOrRd",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.90,
        zorder=5.5,
    )
    colorbar = fig_b.colorbar(scatter, ax=ax_b, fraction=0.046, pad=0.02)
    colorbar.set_label("FreqÃ¼Ã¨ncia de selecciÃ³")
    ax_b.set_title("FreqÃ¼encia de seleccio de parades")
    freq_pdf = save_figure(
        fig_b,
        os.path.join(results_dir, "initial_population_stop_frequency"),
    )

    print("Creant Figura C...")
    fig_c, ax_c = plt.subplots(figsize=(8, 8))
    draw_base_network(ax_c, network, edge_paths)
    sample_records = build_family_sample(ga.population_records, per_family=3)

    for record in sample_records:
        family = str(record["family"])
        individual = list(record["individual"])
        line = build_line_from_stops(network, individual)
        line_edges = [normalize_edge(u, v) for (u, v) in line["line_edges"]]

        for u, v in line_edges:
            points = edge_paths[normalize_edge(u, v)]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax_c.plot(
                xs,
                ys,
                color=family_colors[family],
                linewidth=1.8,
                alpha=0.72,
                zorder=4.5,
            )

        line_stop_nodes = [int(stop) for stop in individual]
        xs_stops = [positions[stop][0] for stop in line_stop_nodes]
        ys_stops = [positions[stop][1] for stop in line_stop_nodes]
        ax_c.scatter(
            xs_stops,
            ys_stops,
            s=62,
            color=family_colors[family],
            edgecolors="white",
            linewidths=0.7,
            alpha=0.88,
            zorder=5.8,
        )

    legend_elements = [
        Line2D([0], [0], color=family_colors["service_best"], lw=2.2, label="Service Best"),
        Line2D([0], [0], color=family_colors["service"], lw=2.2, label="Service"),
        Line2D([0], [0], color=family_colors["demand"], lw=2.2, label="Demand"),
        Line2D([0], [0], color=family_colors["spatial"], lw=2.2, label="Spatial"),
        Line2D([0], [0], color=family_colors["hybrid"], lw=2.2, label="Hybrid"),
        Line2D([0], [0], color=family_colors["random"], lw=2.2, label="Random"),
    ]
    ax_c.legend(handles=legend_elements, loc="upper right", frameon=True)
    ax_c.set_title("Mostra de linies inicials")
    lines_pdf = save_figure(
        fig_c,
        os.path.join(results_dir, "initial_population_sample_lines"),
    )

    print("Creant Figura D...")
    top_service_records = sorted(
        ga.population_records,
        key=lambda record: ga.compute_set_passenger_service(list(record["individual"])),
        reverse=True,
    )[:10]
    fig_d, ax_d = plt.subplots(figsize=(8, 8))
    draw_base_network(ax_d, network, edge_paths)

    for record in top_service_records:
        family = str(record["family"])
        individual = list(record["individual"])
        line = build_line_from_stops(network, individual)
        line_edges = [normalize_edge(u, v) for (u, v) in line["line_edges"]]

        for u, v in line_edges:
            points = edge_paths[normalize_edge(u, v)]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax_d.plot(
                xs,
                ys,
                color=family_colors[family],
                linewidth=2.0,
                alpha=0.62,
                zorder=4.6,
            )

        xs_stops = [positions[stop][0] for stop in individual]
        ys_stops = [positions[stop][1] for stop in individual]
        ax_d.scatter(
            xs_stops,
            ys_stops,
            s=66,
            color=family_colors[family],
            edgecolors="white",
            linewidths=0.75,
            alpha=0.92,
            zorder=5.9,
        )

    ax_d.legend(handles=legend_elements, loc="upper right", frameon=True)
    ax_d.set_title("Top candidats inicials per passenger_service")
    top_service_pdf = save_figure(
        fig_d,
        os.path.join(results_dir, "initial_population_top_service_candidates"),
    )

    print("\n=== RESUM INICIALITZACIO ===")
    print("Mida poblacio:", len(ga.population))
    print("Registres disponibles:", len(ga.population_records))
    print("Top passenger_service candidates:")
    for index, record in enumerate(top_service_records, start=1):
        individual = list(record["individual"])
        family = str(record["family"])
        passenger_service = ga.compute_set_passenger_service(individual)
        print(f" {index:02d}. family={family} passenger_service={passenger_service} individual={individual}")
    print("Fitxers PDF guardats a:")
    print(" -", seed_pdf)
    print(" -", freq_pdf)
    print(" -", lines_pdf)
    print(" -", top_service_pdf)


if __name__ == "__main__":
    main()

