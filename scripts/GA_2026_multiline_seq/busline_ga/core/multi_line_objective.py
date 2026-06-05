from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from busline_ga.core.line_builder import build_line_from_stops
from busline_ga.core.network_model import Edge, NetworkModel, Node
from busline_ga.core.objective_function import (
    EvaluationResult,
    NormalizationBounds,
    ObjectiveFunction,
    RawLineEvaluation,
)


class MultiLineObjectiveFunction(ObjectiveFunction):
    def __init__(
        self,
        network: NetworkModel,
        fixed_lines: Sequence[Sequence[Node]],
        normalization_bounds: Optional[NormalizationBounds] = None,
        normalization_warnings: bool = False,
        shared_stop_penalty: float = 0.0,
    ) -> None:
        super().__init__(
            network=network,
            normalization_warnings=normalization_warnings,
        )
        self.fixed_lines: List[List[Node]] = [list(line) for line in fixed_lines]
        self.fixed_edge_line_counts: Dict[Edge, int] = {}
        self.shared_stop_penalty = shared_stop_penalty
        self.precompute_edge_costs()
        self.precompute_edge_service()

        if normalization_bounds is not None:
            self.set_normalization_bounds(normalization_bounds)

        self.fixed_edge_line_counts = self._build_fixed_edge_line_counts()

    def _line_unique_edges(self, stops: Sequence[Node]) -> List[Edge]:
        line_data = build_line_from_stops(self.network, list(stops))
        unique_edges = list(dict.fromkeys(line_data["line_edges"]))
        result = unique_edges
        return result

    def _build_fixed_edge_line_counts(self) -> Dict[Edge, int]:
        edge_line_counts: Dict[Edge, int] = {}

        for line in self.fixed_lines:
            unique_edges = self._line_unique_edges(line)

            for edge in unique_edges:
                edge_line_counts[edge] = edge_line_counts.get(edge, 0) + 1

        result = edge_line_counts
        return result

    def _candidate_shared_edge_service(self, candidate_edges: Sequence[Edge]) -> float:
        total_service = 0.0
        unique_candidate_edges = list(dict.fromkeys(candidate_edges))

        for edge in unique_candidate_edges:
            base_service = self.edge_service.get(edge, 0.0)
            line_count = self.fixed_edge_line_counts.get(edge, 0) + 1
            total_service += base_service / float(line_count)

        result = total_service
        return result

    def _candidate_shared_stop_ratio(self, stops: Sequence[Node]) -> float:
        fixed_stops = {
            stop
            for line in self.fixed_lines
            for stop in line
        }
        candidate_stops = set(stops)
        shared_count = sum(1 for stop in candidate_stops if stop in fixed_stops)
        ratio = shared_count / max(1, len(candidate_stops))
        result = ratio
        return result

    def _compute_raw_line_evaluation(self, stops: List[Node]) -> RawLineEvaluation:
        line_data = build_line_from_stops(self.network, stops)
        line_edges = line_data["line_edges"]
        line_nodes_real = line_data["line_nodes_real"]
        cost = self.compute_line_cost(line_edges)
        passenger_service = self._candidate_shared_edge_service(line_edges)
        raw_evaluation = RawLineEvaluation(
            cost=cost,
            passenger_service=passenger_service,
            line_edges=line_edges,
            line_nodes_real=line_nodes_real,
        )
        return raw_evaluation

    def _system_objective_components(
        self,
        lines: Sequence[Sequence[Node]],
        lambda_: float,
        target_num_lines: int,
    ) -> Tuple[float, float, float, float, float]:
        total_route_cost = 0.0
        edge_counts: Dict[Edge, int] = {}

        for line in lines:
            line_data = build_line_from_stops(self.network, list(line))
            line_edges = line_data["line_edges"]
            total_route_cost += self.compute_line_cost(line_edges)

            for edge in dict.fromkeys(line_edges):
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        adjusted_passenger_service = sum(
            self.edge_service.get(edge, 0.0) / float(count)
            for edge, count in edge_counts.items()
            if count > 0
        )
        n_lines = max(1, target_num_lines)
        cost_norm = 0.0
        service_norm = 0.0

        if self.normalization_bounds is not None:
            cost_norm = self._min_max_normalize(
                total_route_cost,
                self.normalization_bounds.cost_min * n_lines,
                self.normalization_bounds.cost_max * n_lines,
            )

            if self.normalization_bounds.service_max > 0.0:
                service_norm = (
                    adjusted_passenger_service
                    / (self.normalization_bounds.service_max * n_lines)
                )
        else:
            cost_norm = total_route_cost
            service_norm = adjusted_passenger_service

        fitness = (1.0 - lambda_) * service_norm - lambda_ * cost_norm
        result = (
            fitness,
            adjusted_passenger_service,
            total_route_cost,
            cost_norm,
            service_norm,
        )
        return result

    def evaluate(self, stops: List[Node], lambda_: float) -> EvaluationResult:
        candidate_line = list(stops)
        line_data = build_line_from_stops(self.network, candidate_line)
        line_edges = line_data["line_edges"]
        line_nodes_real = line_data["line_nodes_real"]
        cost = self.compute_line_cost(line_edges)
        target_num_lines = len(self.fixed_lines) + 1
        fixed_components = self._system_objective_components(
            self.fixed_lines,
            lambda_,
            target_num_lines,
        )
        candidate_components = self._system_objective_components(
            [*self.fixed_lines, candidate_line],
            lambda_,
            target_num_lines,
        )
        marginal_fitness = candidate_components[0] - fixed_components[0]
        marginal_passenger_service = candidate_components[1] - fixed_components[1]
        marginal_cost_norm = candidate_components[3] - fixed_components[3]
        marginal_service_norm = candidate_components[4] - fixed_components[4]
        result = EvaluationResult(
            fitness=marginal_fitness,
            cost=cost,
            passenger_service=marginal_passenger_service,
            service=marginal_passenger_service,
            cost_norm=marginal_cost_norm,
            passenger_service_norm=marginal_service_norm,
            service_norm=marginal_service_norm,
            line_edges=line_edges,
            line_nodes_real=line_nodes_real,
        )
        return result
