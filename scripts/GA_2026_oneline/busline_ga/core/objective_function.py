from __future__ import annotations

import math
from math import comb
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from busline_ga.core.line_builder import build_line_from_stops, canonical_edge, path_to_edges
from busline_ga.core.network_model import Edge, NetworkModel, Node

@dataclass
class NormalizationBounds:
    service_min: float
    service_max: float
    cost_min: float
    cost_max: float

@dataclass
class RawLineEvaluation:
    cost: float
    passenger_service: float
    line_edges: List[Edge]
    line_nodes_real: List[Node]

@dataclass
class EvaluationResult:
    fitness: float
    cost: float
    passenger_service: float
    service: float
    cost_norm: float
    passenger_service_norm: float
    service_norm: float
    line_edges: List[Edge]
    line_nodes_real: List[Node]

class ObjectiveFunction:
    def __init__(
        self,
        network: NetworkModel,
        normalization_warnings: bool = False,
    ) -> None:
        self.network = network
        self.normalization_warnings = normalization_warnings
        self.edge_cost: Dict[Edge, float] = {}
        self.edge_service: Dict[Edge, float] = {}
        self.stop_demand_scores: Dict[Node, float] = {}
        self.normalization_bounds: Optional[NormalizationBounds] = None
        self.precompute_stop_demand_scores()

    def set_normalization_bounds(self, bounds: NormalizationBounds) -> None:
        self.normalization_bounds = bounds

    def precompute_edge_costs(self) -> None:
        for u, v, data in self.network.graph.edges(data=True):
            edge = canonical_edge(u, v)
            self.edge_cost[edge] = float(data["cost"])

    def precompute_edge_service(self) -> None:
        self.edge_service = {}
        bus_stops = self.network.bus_stops
        n_stops = len(bus_stops)

        for i in range(n_stops):
            for j in range(i + 1, n_stops):
                stop_i = bus_stops[i]
                stop_j = bus_stops[j]
                demand = self.network.get_od_value(stop_i, stop_j)

                if demand <= 0.0:
                    continue

                shortest_paths = self.network.get_all_shortest_paths_between_stops(stop_i, stop_j)
                sigma_ij = len(shortest_paths)

                if sigma_ij == 0:
                    continue

                l_ij = len(shortest_paths[0]) - 1

                if l_ij <= 0:
                    continue

                for path in shortest_paths:
                    path_edges = path_to_edges(path)

                    for edge in path_edges:
                        contribution = demand * (1.0 / sigma_ij) * (1.0 / l_ij)

                        if edge not in self.edge_service:
                            self.edge_service[edge] = 0.0

                        self.edge_service[edge] += contribution

    def precompute_stop_demand_scores(self) -> None:
        self.stop_demand_scores = {}
        bus_stops = self.network.bus_stops

        for stop in bus_stops:
            total_demand = 0.0
            for other_stop in bus_stops:
                if other_stop == stop:
                    continue
                total_demand += float(self.network.get_od_value(stop, other_stop))
                total_demand += float(self.network.get_od_value(other_stop, stop))
            self.stop_demand_scores[stop] = total_demand

    def compute_line_cost(self, line_edges: List[Edge]) -> float:
        total_cost = 0.0

        for edge in line_edges:
            if edge in self.edge_cost:
                total_cost += self.edge_cost[edge]

        return total_cost

    def compute_line_service(self, line_edges: List[Edge]) -> float:
        total_service = 0.0

        for edge in line_edges:
            if edge in self.edge_service:
                total_service += self.edge_service[edge]

        return total_service

    def compute_stop_service(self, selected_stops: List[Node]) -> float:
        total_service = 0.0

        for stop in selected_stops:
            total_service += self.stop_demand_scores.get(stop, 0.0)

        return total_service

    def compute_passenger_service(self, selected_stops: List[Node]) -> float:
        total_service = 0.0

        for i in range(len(selected_stops)):
            for j in range(i + 1, len(selected_stops)):
                stop_i = selected_stops[i]
                stop_j = selected_stops[j]
                total_service += float(self.network.get_od_value(stop_i, stop_j))

        result = total_service
        return result

    def _compute_raw_line_evaluation(self, stops: List[Node]) -> RawLineEvaluation:
        line_data = build_line_from_stops(self.network, stops)
        line_edges = line_data["line_edges"]
        line_nodes_real = line_data["line_nodes_real"]
        cost = self.compute_line_cost(line_edges)
        passenger_service = self.compute_passenger_service(stops)
        raw_evaluation = RawLineEvaluation(
            cost=cost,
            passenger_service=passenger_service,
            line_edges=line_edges,
            line_nodes_real=line_nodes_real,
        )
        return raw_evaluation

    def _min_max_normalize(self, value: float, minimum: float, maximum: float) -> float:
        normalized = 0.0

        if maximum > minimum:
            normalized = (value - minimum) / (maximum - minimum)

        return normalized

    def _warn_if_outside_normalization_range(
        self,
        name: str,
        value: float,
        normalized: float,
        minimum: float,
        maximum: float,
    ) -> None:
        if self.normalization_warnings and (normalized < 0.0 or normalized > 1.0):
            print(
                f"[NORMALIZATION WARNING] {name}: "
                f"value={value:.6f}, min={minimum:.6f}, max={maximum:.6f}, "
                f"normalized={normalized:.6f}"
            )

    def estimate_normalization_bounds(
        self,
        candidate_lines: Sequence[List[Node]],
    ) -> NormalizationBounds:
        raw_evaluations = [
            self._compute_raw_line_evaluation(candidate_line)
            for candidate_line in candidate_lines
        ]

        if raw_evaluations:
            service_values = [evaluation.passenger_service for evaluation in raw_evaluations]
            cost_values = [evaluation.cost for evaluation in raw_evaluations]
            bounds = NormalizationBounds(
                service_min=0.0,
                service_max=max(service_values),
                cost_min=min(cost_values),
                cost_max=max(cost_values),
            )
        else:
            bounds = NormalizationBounds(
                service_min=0.0,
                service_max=0.0,
                cost_min=0.0,
                cost_max=0.0,
            )

        return bounds

    def estimate_passenger_service_upper_bound_for_fixed_length(
        self,
        line_length: int,
    ) -> float:
        pair_count = comb(line_length, 2)
        service_values: List[float] = []

        for i in range(len(self.network.bus_stops)):
            for j in range(i + 1, len(self.network.bus_stops)):
                stop_i = self.network.bus_stops[i]
                stop_j = self.network.bus_stops[j]
                service_values.append(float(self.network.get_od_value(stop_i, stop_j)))

        service_values.sort(reverse=True)
        service_max = sum(service_values[:pair_count])
        result = service_max
        return result

    def estimate_cost_bounds_for_fixed_length(
        self,
        line_length: int,
    ) -> Tuple[float, float]:
        transition_count = max(0, line_length - 1)
        cost_values: List[float] = []

        for i in range(len(self.network.bus_stops)):
            for j in range(i + 1, len(self.network.bus_stops)):
                stop_i = self.network.bus_stops[i]
                stop_j = self.network.bus_stops[j]
                cost = float(self.network.get_shortest_cost_between_stops(stop_i, stop_j))

                if math.isfinite(cost):
                    cost_values.append(cost)

        cost_values.sort()
        effective_count = min(transition_count, len(cost_values))

        if effective_count == 0:
            cost_min = 0.0
            cost_max = 0.0
        else:
            cost_min = sum(cost_values[:effective_count])
            cost_max = sum(cost_values[-effective_count:])

        result = (cost_min, cost_max)
        return result

    def estimate_structural_bounds_for_fixed_length(
        self,
        line_length: int,
    ) -> NormalizationBounds:
        cost_min, cost_max = self.estimate_cost_bounds_for_fixed_length(line_length)
        service_max = self.estimate_passenger_service_upper_bound_for_fixed_length(line_length)
        bounds = NormalizationBounds(
            service_min=0.0,
            service_max=service_max,
            cost_min=cost_min,
            cost_max=cost_max,
        )
        return bounds

    def _unordered_stop_pair_values(self) -> Tuple[List[float], List[float]]:
        service_values: List[float] = []
        cost_values: List[float] = []
        bus_stops = self.network.bus_stops

        for i in range(len(bus_stops)):
            for j in range(i + 1, len(bus_stops)):
                stop_i = bus_stops[i]
                stop_j = bus_stops[j]
                service_values.append(self.network.get_od_value(stop_i, stop_j))
                cost_values.append(self.network.get_shortest_cost_between_stops(stop_i, stop_j))

        return service_values, cost_values

    def estimate_combinatorial_bounds_for_fixed_length(
        self,
        line_length: int,
    ) -> NormalizationBounds:
        pair_count = comb(line_length, 2)
        service_values, cost_values = self._unordered_stop_pair_values()
        service_sorted = sorted(service_values)
        cost_sorted = sorted(cost_values)
        effective_pair_count = min(pair_count, len(service_sorted), len(cost_sorted))

        if effective_pair_count == 0:
            bounds = NormalizationBounds(
                service_min=0.0,
                service_max=0.0,
                cost_min=0.0,
                cost_max=0.0,
            )
        else:
            bounds = NormalizationBounds(
                service_min=0.0,
                service_max=sum(service_sorted[-effective_pair_count:]),
                cost_min=sum(cost_sorted[:effective_pair_count]),
                cost_max=sum(cost_sorted[-effective_pair_count:]),
            )

        return bounds

    def evaluate(self, stops: List[Node], lambda_: float) -> EvaluationResult:
        raw_evaluation = self._compute_raw_line_evaluation(stops)
        cost_norm = raw_evaluation.cost
        passenger_service_norm = raw_evaluation.passenger_service

        if self.normalization_bounds is not None:
            cost_norm = self._min_max_normalize(
                raw_evaluation.cost,
                self.normalization_bounds.cost_min,
                self.normalization_bounds.cost_max,
            )
            self._warn_if_outside_normalization_range(
                "cost_norm",
                raw_evaluation.cost,
                cost_norm,
                self.normalization_bounds.cost_min,
                self.normalization_bounds.cost_max,
            )

            passenger_service_norm = 0.0
            if self.normalization_bounds.service_max > 0.0:
                passenger_service_norm = (
                    raw_evaluation.passenger_service / self.normalization_bounds.service_max
                )
            elif self.normalization_warnings:
                print("[NORMALIZATION WARNING] passenger_service_max is zero.")

            self._warn_if_outside_normalization_range(
                "passenger_service_norm",
                raw_evaluation.passenger_service,
                passenger_service_norm,
                0.0,
                self.normalization_bounds.service_max,
            )

        fitness = (1.0 - lambda_) * passenger_service_norm - lambda_ * cost_norm
        result = EvaluationResult(
            fitness=fitness,
            cost=raw_evaluation.cost,
            passenger_service=raw_evaluation.passenger_service,
            service=raw_evaluation.passenger_service,
            cost_norm=cost_norm,
            passenger_service_norm=passenger_service_norm,
            service_norm=passenger_service_norm,
            line_edges=raw_evaluation.line_edges,
            line_nodes_real=raw_evaluation.line_nodes_real,
        )
        return result

