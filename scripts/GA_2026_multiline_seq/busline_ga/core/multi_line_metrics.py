from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from busline_ga.core.network_model import Edge, NetworkModel
from busline_ga.core.objective_function import EvaluationResult, NormalizationBounds, ObjectiveFunction


def _normalize_value(value: float, minimum: float, maximum: float) -> float:
    normalized = 0.0

    if maximum > minimum:
        normalized = (value - minimum) / (maximum - minimum)

    return normalized


def _build_objective_for_metrics(
    network: NetworkModel,
    normalization_bounds: Optional[NormalizationBounds],
) -> ObjectiveFunction:
    objective = ObjectiveFunction(network=network)
    objective.precompute_edge_costs()
    objective.precompute_edge_service()

    if normalization_bounds is not None:
        objective.set_normalization_bounds(normalization_bounds)

    return objective


def _unique_edges_per_line(evaluations: Sequence[EvaluationResult]) -> List[Sequence[Edge]]:
    return [list(dict.fromkeys(result.line_edges)) for result in evaluations]


def _unique_stops_per_line(lines: Sequence[Sequence[int]]) -> List[Sequence[int]]:
    return [list(dict.fromkeys(line)) for line in lines]


def compute_pairwise_line_overlap(
    lines: Sequence[Sequence[int]],
    evaluations: Sequence[EvaluationResult],
) -> List[Dict[str, int]]:
    overlap: List[Dict[str, int]] = []
    stop_sets = [set(line) for line in lines]
    edge_sets = [set(result.line_edges) for result in evaluations]

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            overlap.append(
                {
                    "line_i": i + 1,
                    "line_j": j + 1,
                    "shared_stops": len(stop_sets[i].intersection(stop_sets[j])),
                    "shared_edges": len(edge_sets[i].intersection(edge_sets[j])),
                }
            )

    return overlap


def compute_adjusted_line_metrics(
    lines: Sequence[Sequence[int]],
    evaluations: Sequence[EvaluationResult],
    network: NetworkModel,
    normalization_bounds: Optional[NormalizationBounds],
    lambda_: float,
) -> List[Dict[str, float]]:
    objective = _build_objective_for_metrics(network, normalization_bounds)
    unique_edges_by_line = _unique_edges_per_line(evaluations)

    edge_counts: Counter[Edge] = Counter()
    for unique_edges in unique_edges_by_line:
        for edge in unique_edges:
            edge_counts[edge] += 1

    adjusted_metrics: List[Dict[str, float]] = []

    for index, evaluation in enumerate(evaluations, start=1):
        unique_edges = unique_edges_by_line[index - 1]
        adjusted_passenger_service = sum(
            objective.edge_service.get(edge, 0.0) / float(edge_counts[edge])
            for edge in unique_edges
            if edge_counts[edge] > 0
        )
        cost_norm = evaluation.cost
        adjusted_service_norm = adjusted_passenger_service

        if normalization_bounds is not None:
            cost_norm = _normalize_value(
                evaluation.cost,
                normalization_bounds.cost_min,
                normalization_bounds.cost_max,
            )
            adjusted_service_norm = 0.0

            if normalization_bounds.service_max > 0.0:
                adjusted_service_norm = adjusted_passenger_service / normalization_bounds.service_max

        adjusted_fitness = (1.0 - lambda_) * adjusted_service_norm - lambda_ * cost_norm
        shared_edge_count = sum(1 for edge in unique_edges if edge_counts[edge] > 1)
        shared_edge_ratio = shared_edge_count / len(unique_edges) if unique_edges else 0.0

        adjusted_metrics.append(
            {
                "line_index": index,
                "original_fitness": evaluation.fitness,
                "original_passenger_service": evaluation.passenger_service,
                "adjusted_passenger_service": adjusted_passenger_service,
                "cost": evaluation.cost,
                "cost_norm": cost_norm,
                "adjusted_service_norm": adjusted_service_norm,
                "adjusted_fitness": adjusted_fitness,
                "shared_edge_count_for_line": shared_edge_count,
                "shared_edge_ratio_for_line": shared_edge_ratio,
            }
        )

    return adjusted_metrics


def compute_multiline_system_metrics(
    lines: Sequence[Sequence[int]],
    evaluations: Sequence[EvaluationResult],
    network: NetworkModel,
    normalization_bounds: Optional[NormalizationBounds],
    lambda_: float,
) -> Dict[str, float]:
    objective = _build_objective_for_metrics(network, normalization_bounds)

    total_route_cost = sum(result.cost for result in evaluations)
    naive_passenger_service = sum(result.passenger_service for result in evaluations)

    edge_counts: Counter[Edge] = Counter()
    for result in evaluations:
        unique_edges = set(result.line_edges)
        for edge in unique_edges:
            edge_counts[edge] += 1

    unique_edges = list(edge_counts.keys())
    unique_edge_cost = sum(objective.edge_cost.get(edge, 0.0) for edge in unique_edges)
    adjusted_passenger_service = sum(
        objective.edge_service.get(edge, 0.0) / count
        for edge, count in edge_counts.items()
    )

    total_edge_occurrences = sum(len(set(result.line_edges)) for result in evaluations)
    unique_edge_count = len(unique_edges)
    shared_edge_count = sum(1 for count in edge_counts.values() if count > 1)
    shared_edge_ratio = (
        shared_edge_count / unique_edge_count if unique_edge_count > 0 else 0.0
    )

    stop_counts: Counter[int] = Counter()
    for line in lines:
        for stop in set(line):
            stop_counts[stop] += 1

    unique_stop_count = len(stop_counts)
    total_stop_occurrences = sum(len(line) for line in lines)
    shared_stop_count = sum(1 for count in stop_counts.values() if count > 1)
    shared_stop_ratio = (
        shared_stop_count / unique_stop_count if unique_stop_count > 0 else 0.0
    )

    service_per_total_cost = (
        adjusted_passenger_service / total_route_cost if total_route_cost > 0.0 else 0.0
    )
    service_per_unique_edge_cost = (
        adjusted_passenger_service / unique_edge_cost if unique_edge_cost > 0.0 else 0.0
    )

    system_cost_norm = 0.0
    system_service_norm = 0.0

    if normalization_bounds is not None:
        system_cost_norm = _normalize_value(
            total_route_cost,
            normalization_bounds.cost_min * len(lines),
            normalization_bounds.cost_max * len(lines),
        )
        if normalization_bounds.service_max > 0.0:
            system_service_norm = (
                adjusted_passenger_service
                / (normalization_bounds.service_max * len(lines))
            )

    system_fitness = (
        (1.0 - lambda_) * system_service_norm - lambda_ * system_cost_norm
    )

    return {
        "total_route_cost": total_route_cost,
        "unique_edge_cost": unique_edge_cost,
        "adjusted_passenger_service": adjusted_passenger_service,
        "naive_passenger_service": naive_passenger_service,
        "service_per_total_cost": service_per_total_cost,
        "service_per_unique_edge_cost": service_per_unique_edge_cost,
        "system_cost_norm": system_cost_norm,
        "system_service_norm": system_service_norm,
        "system_fitness": system_fitness,
        "unique_stop_count": unique_stop_count,
        "total_stop_occurrences": total_stop_occurrences,
        "shared_stop_count": shared_stop_count,
        "shared_stop_ratio": shared_stop_ratio,
        "unique_edge_count": unique_edge_count,
        "total_edge_occurrences": total_edge_occurrences,
        "shared_edge_count": shared_edge_count,
        "shared_edge_ratio": shared_edge_ratio,
    }
