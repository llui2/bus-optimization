from __future__ import annotations

import os
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from busline_ga.core.line_builder import path_to_edges
from busline_ga.core.network_model import Edge
from busline_ga.core.multi_line_metrics import (
    compute_adjusted_line_metrics,
    compute_pairwise_line_overlap,
)
from busline_ga.core.network_model import NetworkModel, Node
from busline_ga.core.objective_function import EvaluationResult, NormalizationBounds, ObjectiveFunction


Line = List[Node]
SystemIndividual = List[Line]


def _filesystem_path(path: str) -> str:
    filesystem_path = path

    if os.name == "nt":
        absolute_path = os.path.abspath(path)
        if not absolute_path.startswith("\\\\?\\"):
            filesystem_path = "\\\\?\\" + absolute_path

    return filesystem_path


@dataclass
class JointEvaluationResult:
    fitness: float
    system_metrics: Dict[str, float]
    adjusted_line_metrics: List[Dict[str, float]]
    pairwise_overlap: List[Dict[str, int]]
    lines: SystemIndividual
    evaluations: List[EvaluationResult]
    total_route_cost: float
    adjusted_passenger_service: float
    system_cost_norm: float
    system_service_norm: float
    shared_stop_ratio: float
    shared_edge_ratio: float


class JointMultiLineOptimizer:
    """
    Joint/parallel multi-line GA.

    Sequential GA optimizes and fixes lines one after another. This optimizer
    evolves all lines simultaneously, so one GA individual is a complete bus
    system: List[List[Node]].
    """

    def __init__(
        self,
        network: NetworkModel,
        n_lines: int,
        line_length: int,
        population_size: int,
        generations: int,
        lambda_: float,
        seed: int,
        normalization_bounds: NormalizationBounds,
        mutation_prob: float = 0.20,
        crossover_prob: float = 0.80,
        n_elite: int = 2,
        tournament_size: int = 3,
        init_ratio_service: float = 0.0,
        init_ratio_demand: float = 0.0,
        init_ratio_spatial: float = 0.0,
        init_ratio_hybrid: float = 0.0,
        init_ratio_random: float = 1.0,
        inject_best_service_candidate: bool = False,
        compactness_penalty: float = 0.0,
        max_segment_cost_soft: Optional[float] = None,
        max_line_cost_soft: Optional[float] = None,
        max_edges_per_line_soft: Optional[float] = None,
        save_improvement_snapshots: bool = False,
        snapshot_output_dir: Optional[str] = None,
        save_evolution_history: bool = False,
        evolution_output_dir: Optional[str] = None,
        od_case: Optional[str] = None,
        map_case: Optional[str] = None,
    ) -> None:
        self.network = network
        self.n_lines = n_lines
        self.line_length = line_length
        self.population_size = population_size
        self.generations = generations
        self.lambda_ = lambda_
        self.seed = seed
        self.normalization_bounds = normalization_bounds
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.n_elite = n_elite
        self.tournament_size = tournament_size
        self.init_ratio_service = init_ratio_service
        self.init_ratio_demand = init_ratio_demand
        self.init_ratio_spatial = init_ratio_spatial
        self.init_ratio_hybrid = init_ratio_hybrid
        self.init_ratio_random = init_ratio_random
        self.inject_best_service_candidate = inject_best_service_candidate
        self.compactness_penalty = compactness_penalty
        self.max_segment_cost_soft = max_segment_cost_soft
        self.max_line_cost_soft = max_line_cost_soft
        self.max_edges_per_line_soft = max_edges_per_line_soft
        self.save_improvement_snapshots = save_improvement_snapshots
        self.snapshot_output_dir = snapshot_output_dir
        self.save_evolution_history = save_evolution_history
        self.evolution_output_dir = evolution_output_dir
        self.od_case = od_case
        self.map_case = map_case
        self.rng = random.Random(seed)
        self.population: List[SystemIndividual] = []
        self.evolution_history: List[Dict[str, Any]] = []
        self.improvement_history: List[Dict[str, Any]] = []
        self._max_line_edge_count_bound_cache: Optional[int] = None
        self._system_service_max_cache: Dict[int, float] = {}
        self.objective = ObjectiveFunction(network=self.network)
        self.objective.precompute_edge_costs()
        self.objective.precompute_edge_service()
        self.objective.set_normalization_bounds(self.normalization_bounds)

    def _normalize_value(self, value: float, minimum: float, maximum: float) -> float:
        normalized = 0.0

        if maximum > minimum:
            normalized = (value - minimum) / (maximum - minimum)

        return normalized

    def _clip_normalized_value(self, value: float) -> float:
        clipped = max(0.0, min(1.0, value))
        return clipped

    def _single_line_service_min(self) -> float:
        result = self.normalization_bounds.service_min
        return result

    def _single_line_service_max(self) -> float:
        result = self.normalization_bounds.service_max
        return result

    def _single_line_cost_min(self) -> float:
        result = self.normalization_bounds.cost_min
        return result

    def _single_line_cost_max(self) -> float:
        result = self.normalization_bounds.cost_max
        return result

    def _system_cost_min(self, n_lines: int) -> float:
        result = self._single_line_cost_min() * n_lines
        return result

    def _system_cost_max(self, n_lines: int) -> float:
        result = self._single_line_cost_max() * n_lines
        return result

    def _max_line_edge_count_bound(self) -> int:
        if self._max_line_edge_count_bound_cache is None:
            transition_count = max(0, self.line_length - 1)
            edge_counts: List[int] = []
            stops = list(self.network.bus_stops)

            for i in range(len(stops)):
                for j in range(i + 1, len(stops)):
                    path = self.network.get_shortest_path_between_stops(stops[i], stops[j])
                    edge_counts.append(max(0, len(path) - 1))

            edge_counts.sort(reverse=True)
            self._max_line_edge_count_bound_cache = (
                sum(edge_counts[:transition_count]) if transition_count > 0 else 0
            )

        result = self._max_line_edge_count_bound_cache
        return result

    def _system_service_min(self) -> float:
        result = 0.0
        return result

    def _system_service_max(self, n_lines: int) -> float:
        if n_lines not in self._system_service_max_cache:
            edge_service_values = sorted(self.objective.edge_service.values(), reverse=True)
            max_unique_edges = max(0, n_lines * self._max_line_edge_count_bound())
            effective_count = min(len(edge_service_values), max_unique_edges)
            service_max = sum(edge_service_values[:effective_count]) if effective_count > 0 else 0.0
            self._system_service_max_cache[n_lines] = max(
                service_max,
                self._single_line_service_max() * n_lines,
            )

        result = self._system_service_max_cache[n_lines]
        return result

    def _positive_excess_ratio(self, value: float, soft_limit: float) -> float:
        excess_ratio = 0.0

        if soft_limit > 0.0:
            excess_ratio = max(0.0, value - soft_limit) / soft_limit

        return excess_ratio

    def _resolve_max_line_cost_soft(self) -> float:
        cost_range = max(0.0, self.normalization_bounds.cost_max - self.normalization_bounds.cost_min)
        soft_limit = self.normalization_bounds.cost_min + 0.25 * cost_range

        if self.max_line_cost_soft is not None:
            soft_limit = self.max_line_cost_soft

        result = max(0.0, soft_limit)
        return result

    def _resolve_max_segment_cost_soft(self) -> float:
        segment_count = max(1, self.line_length - 1)
        soft_limit = 0.65 * self.normalization_bounds.cost_max / segment_count

        if self.max_segment_cost_soft is not None:
            soft_limit = self.max_segment_cost_soft

        result = max(0.0, soft_limit)
        return result

    def _resolve_max_edges_per_line_soft(self) -> float:
        soft_limit = float(max(1, self.line_length - 1) * 8)

        if self.max_edges_per_line_soft is not None:
            soft_limit = self.max_edges_per_line_soft

        result = max(0.0, soft_limit)
        return result

    def _copy_system(self, system: SystemIndividual) -> SystemIndividual:
        result = [list(line) for line in system]
        return result

    def _random_line(self) -> Line:
        stops = list(self.network.bus_stops)
        if len(stops) >= self.line_length:
            line = self.rng.sample(stops, self.line_length)
        else:
            line = [self.rng.choice(stops) for _ in range(self.line_length)]
            line = self._repair_line(line)
        return line

    def _repair_line(self, line: Sequence[Node]) -> Line:
        stops = list(self.network.bus_stops)
        repaired: Line = []
        used = set()

        for stop in line:
            if stop not in used and stop in stops:
                repaired.append(stop)
                used.add(stop)

        available = [stop for stop in stops if stop not in used]
        while len(repaired) < self.line_length and available:
            stop = self.rng.choice(available)
            repaired.append(stop)
            used.add(stop)
            available = [candidate for candidate in available if candidate != stop]

        while len(repaired) < self.line_length:
            repaired.append(self.rng.choice(stops))

        result = repaired[:self.line_length]
        return result

    def _repair_system(self, system: Sequence[Sequence[Node]]) -> SystemIndividual:
        repaired = [self._repair_line(line) for line in system[:self.n_lines]]

        while len(repaired) < self.n_lines:
            repaired.append(self._random_line())

        result = repaired[:self.n_lines]
        return result

    def generate_initial_population(self) -> None:
        population: List[SystemIndividual] = []

        for _ in range(self.population_size):
            system = [self._random_line() for _ in range(self.n_lines)]
            population.append(system)

        self.population = population

    def _evaluate_lines(self, system: SystemIndividual) -> List[EvaluationResult]:
        evaluations = [
            self.objective.evaluate(list(line), self.lambda_)
            for line in system
        ]
        return evaluations

    def _compute_compactness_metrics(
        self,
        lines: SystemIndividual,
        evaluations: Sequence[EvaluationResult],
        system_cost_norm: float,
    ) -> Dict[str, float]:
        line_costs = [result.cost for result in evaluations]
        line_edge_counts = [len(result.line_edges) for result in evaluations]
        segment_costs: List[float] = []
        segment_edge_counts: List[int] = []

        for line in lines:
            for index in range(len(line) - 1):
                start_stop = line[index]
                end_stop = line[index + 1]
                path = self.network.get_shortest_path_between_stops(start_stop, end_stop)
                segment_edges = path_to_edges(path)
                segment_cost = sum(
                    self.objective.edge_cost.get(edge, 0.0)
                    for edge in segment_edges
                )
                segment_costs.append(segment_cost)
                segment_edge_counts.append(len(segment_edges))

        average_line_cost = sum(line_costs) / len(line_costs) if line_costs else 0.0
        max_line_cost = max(line_costs) if line_costs else 0.0
        average_segment_cost = (
            sum(segment_costs) / len(segment_costs) if segment_costs else 0.0
        )
        max_segment_cost = max(segment_costs) if segment_costs else 0.0
        average_edges_per_line = (
            sum(line_edge_counts) / len(line_edge_counts) if line_edge_counts else 0.0
        )
        max_edges_per_line = max(line_edge_counts) if line_edge_counts else 0
        average_segment_edges = (
            sum(segment_edge_counts) / len(segment_edge_counts) if segment_edge_counts else 0.0
        )
        max_segment_edges = max(segment_edge_counts) if segment_edge_counts else 0
        max_line_cost_soft = self._resolve_max_line_cost_soft()
        max_segment_cost_soft = self._resolve_max_segment_cost_soft()
        max_edges_per_line_soft = self._resolve_max_edges_per_line_soft()
        compactness_total_cost_component = system_cost_norm
        compactness_max_line_excess = self._positive_excess_ratio(
            max_line_cost,
            max_line_cost_soft,
        )
        compactness_max_segment_excess = self._positive_excess_ratio(
            max_segment_cost,
            max_segment_cost_soft,
        )
        compactness_max_edges_excess = self._positive_excess_ratio(
            float(max_edges_per_line),
            max_edges_per_line_soft,
        )
        compactness_penalty_value = self.compactness_penalty * (
            0.40 * compactness_total_cost_component
            + 0.25 * compactness_max_line_excess
            + 0.25 * compactness_max_segment_excess
            + 0.10 * compactness_max_edges_excess
        )
        result = {
            "compactness_penalty": self.compactness_penalty,
            "max_line_cost_soft": max_line_cost_soft,
            "max_segment_cost_soft": max_segment_cost_soft,
            "max_edges_per_line_soft": max_edges_per_line_soft,
            "compactness_total_cost_component": compactness_total_cost_component,
            "compactness_max_line_excess": compactness_max_line_excess,
            "compactness_max_segment_excess": compactness_max_segment_excess,
            "compactness_max_edges_excess": compactness_max_edges_excess,
            "compactness_penalty_value": compactness_penalty_value,
            "average_line_cost": average_line_cost,
            "max_line_cost": max_line_cost,
            "average_segment_cost": average_segment_cost,
            "max_segment_cost": max_segment_cost,
            "average_edges_per_line": average_edges_per_line,
            "max_edges_per_line": float(max_edges_per_line),
            "average_segment_edges": average_segment_edges,
            "max_segment_edges": float(max_segment_edges),
        }
        return result

    def _compute_system_metrics(
        self,
        lines: SystemIndividual,
        evaluations: Sequence[EvaluationResult],
    ) -> Dict[str, float]:
        total_route_cost = sum(result.cost for result in evaluations)
        raw_line_passenger_service_sum = sum(result.passenger_service for result in evaluations)

        edge_counts: Counter[Edge] = Counter()
        total_route_edges = 0
        internal_repeated_edges = 0
        for result in evaluations:
            line_edges = list(result.line_edges)
            unique_line_edges = set(line_edges)
            total_route_edges += len(line_edges)
            internal_repeated_edges += max(0, len(line_edges) - len(unique_line_edges))
            for edge in unique_line_edges:
                edge_counts[edge] += 1

        unique_edges = list(edge_counts.keys())
        unique_edge_cost = sum(self.objective.edge_cost.get(edge, 0.0) for edge in unique_edges)
        unique_edge_passenger_service = sum(
            self.objective.edge_service.get(edge, 0.0)
            for edge in unique_edges
        )
        naive_edge_passenger_service = sum(
            self.objective.edge_service.get(edge, 0.0) * count
            for edge, count in edge_counts.items()
        )
        adjusted_passenger_service = sum(
            self.objective.edge_service.get(edge, 0.0) / count
            for edge, count in edge_counts.items()
        )
        total_edge_occurrences = sum(len(set(result.line_edges)) for result in evaluations)
        unique_edge_count = len(unique_edges)
        shared_edge_count = sum(1 for count in edge_counts.values() if count > 1)
        shared_edge_ratio = shared_edge_count / unique_edge_count if unique_edge_count > 0 else 0.0
        cross_line_shared_edges = shared_edge_count

        stop_counts: Counter[int] = Counter()
        for line in lines:
            for stop in set(line):
                stop_counts[stop] += 1

        unique_stop_count = len(stop_counts)
        total_stop_occurrences = sum(len(line) for line in lines)
        shared_stop_count = sum(1 for count in stop_counts.values() if count > 1)
        shared_stop_ratio = shared_stop_count / unique_stop_count if unique_stop_count > 0 else 0.0
        service_per_total_cost = adjusted_passenger_service / total_route_cost if total_route_cost > 0.0 else 0.0
        service_per_unique_edge_cost = adjusted_passenger_service / unique_edge_cost if unique_edge_cost > 0.0 else 0.0
        n_lines = len(lines)
        single_line_service_min = self._single_line_service_min()
        single_line_service_max = self._single_line_service_max()
        single_line_cost_min = self._single_line_cost_min()
        single_line_cost_max = self._single_line_cost_max()
        system_service_min = self._system_service_min()
        system_service_max = self._system_service_max(n_lines)
        system_cost_min = self._system_cost_min(n_lines)
        system_cost_max = self._system_cost_max(n_lines)
        system_cost_norm = self._normalize_value(
            total_route_cost,
            system_cost_min,
            system_cost_max,
        )
        system_service_norm = 0.0

        if system_service_max > system_service_min:
            system_service_norm = self._normalize_value(
                adjusted_passenger_service,
                system_service_min,
                system_service_max,
            )

        compactness_metrics = self._compute_compactness_metrics(
            lines,
            evaluations,
            system_cost_norm,
        )
        base_system_fitness = (1.0 - self.lambda_) * system_service_norm - self.lambda_ * system_cost_norm
        system_fitness = base_system_fitness - compactness_metrics["compactness_penalty_value"]
        result = {
            "total_route_cost": total_route_cost,
            "unique_edge_cost": unique_edge_cost,
            "adjusted_passenger_service": adjusted_passenger_service,
            "adjusted_service": adjusted_passenger_service,
            "raw_line_passenger_service_sum": raw_line_passenger_service_sum,
            "raw_route_service_sum": naive_edge_passenger_service,
            "naive_edge_passenger_service": naive_edge_passenger_service,
            "unique_edge_passenger_service": unique_edge_passenger_service,
            "unique_edge_service_sum": unique_edge_passenger_service,
            "single_line_service_min": single_line_service_min,
            "single_line_service_max": single_line_service_max,
            "system_service_min": system_service_min,
            "system_service_max": system_service_max,
            "single_line_cost_min": single_line_cost_min,
            "single_line_cost_max": single_line_cost_max,
            "system_cost_min": system_cost_min,
            "system_cost_max": system_cost_max,
            "service_per_total_cost": service_per_total_cost,
            "service_per_unique_edge_cost": service_per_unique_edge_cost,
            "system_cost_norm": system_cost_norm,
            "system_service_norm": system_service_norm,
            "base_system_fitness": base_system_fitness,
            "system_fitness": system_fitness,
            **compactness_metrics,
            "unique_stop_count": unique_stop_count,
            "total_stop_occurrences": total_stop_occurrences,
            "shared_stop_count": shared_stop_count,
            "shared_stops": float(shared_stop_count),
            "shared_stop_ratio": shared_stop_ratio,
            "unique_edge_count": unique_edge_count,
            "unique_edges": float(unique_edge_count),
            "total_route_edges": float(total_route_edges),
            "total_edge_occurrences": total_edge_occurrences,
            "internal_repeated_edges": float(internal_repeated_edges),
            "cross_line_shared_edges": float(cross_line_shared_edges),
            "shared_edge_count": shared_edge_count,
            "shared_edges": float(shared_edge_count),
            "total_cost": total_route_cost,
            "edge_overlap_ratio": shared_edge_ratio,
            "stop_overlap_ratio": shared_stop_ratio,
            "shared_edge_ratio": shared_edge_ratio,
        }
        return result

    def evaluate_system(
        self,
        system: SystemIndividual,
        include_details: bool = False,
    ) -> JointEvaluationResult:
        repaired_system = self._repair_system(system)
        evaluations = self._evaluate_lines(repaired_system)
        system_metrics = self._compute_system_metrics(repaired_system, evaluations)
        adjusted_line_metrics: List[Dict[str, float]] = []
        pairwise_overlap: List[Dict[str, int]] = []

        if include_details:
            adjusted_line_metrics = compute_adjusted_line_metrics(
                lines=repaired_system,
                evaluations=evaluations,
                network=self.network,
                normalization_bounds=self.normalization_bounds,
                lambda_=self.lambda_,
            )
            pairwise_overlap = compute_pairwise_line_overlap(
                lines=repaired_system,
                evaluations=evaluations,
            )
        result = JointEvaluationResult(
            fitness=system_metrics["system_fitness"],
            system_metrics=system_metrics,
            adjusted_line_metrics=adjusted_line_metrics,
            pairwise_overlap=pairwise_overlap,
            lines=repaired_system,
            evaluations=evaluations,
            total_route_cost=system_metrics["total_route_cost"],
            adjusted_passenger_service=system_metrics["adjusted_passenger_service"],
            system_cost_norm=system_metrics["system_cost_norm"],
            system_service_norm=system_metrics["system_service_norm"],
            shared_stop_ratio=system_metrics["shared_stop_ratio"],
            shared_edge_ratio=system_metrics["shared_edge_ratio"],
        )
        return result

    def _evaluate_population(self) -> List[Tuple[SystemIndividual, JointEvaluationResult]]:
        scored = [
            (self._copy_system(system), self.evaluate_system(system))
            for system in self.population
        ]
        return scored

    def _tournament_select(
        self,
        scored_population: Sequence[Tuple[SystemIndividual, JointEvaluationResult]],
    ) -> SystemIndividual:
        tournament_size = min(self.tournament_size, len(scored_population))
        competitors = self.rng.sample(list(scored_population), tournament_size)
        winner = max(competitors, key=lambda item: item[1].fitness)
        result = self._copy_system(winner[0])
        return result

    def _crossover(self, parent_a: SystemIndividual, parent_b: SystemIndividual) -> SystemIndividual:
        if self.rng.random() >= self.crossover_prob:
            child = self._copy_system(parent_a)
        else:
            child = []
            for line_index in range(self.n_lines):
                source = parent_a if self.rng.random() < 0.5 else parent_b
                child.append(list(source[line_index]))
        result = self._repair_system(child)
        return result

    def _mutate(self, system: SystemIndividual) -> SystemIndividual:
        mutated = self._copy_system(system)

        if self.rng.random() < self.mutation_prob:
            line_index = self.rng.randrange(self.n_lines)
            if self.rng.random() < 0.15:
                mutated[line_index] = self._random_line()
            else:
                stop_index = self.rng.randrange(self.line_length)
                current_line = list(mutated[line_index])
                available = [
                    stop for stop in self.network.bus_stops
                    if stop not in set(current_line) or stop == current_line[stop_index]
                ]
                replacement = self.rng.choice(available)
                current_line[stop_index] = replacement
                mutated[line_index] = self._repair_line(current_line)

        result = self._repair_system(mutated)
        return result

    def _record_generation(self, generation: int, evaluation: JointEvaluationResult) -> None:
        if self.save_evolution_history:
            self.evolution_history.append(
                {
                    "generation": generation,
                    "fitness": evaluation.fitness,
                    "base_system_fitness": evaluation.system_metrics["base_system_fitness"],
                    "compactness_penalty": evaluation.system_metrics["compactness_penalty"],
                    "compactness_penalty_value": evaluation.system_metrics[
                        "compactness_penalty_value"
                    ],
                    "compactness_total_cost_component": evaluation.system_metrics[
                        "compactness_total_cost_component"
                    ],
                    "compactness_max_line_excess": evaluation.system_metrics[
                        "compactness_max_line_excess"
                    ],
                    "compactness_max_segment_excess": evaluation.system_metrics[
                        "compactness_max_segment_excess"
                    ],
                    "compactness_max_edges_excess": evaluation.system_metrics[
                        "compactness_max_edges_excess"
                    ],
                    "adjusted_passenger_service": evaluation.adjusted_passenger_service,
                    "raw_line_passenger_service_sum": evaluation.system_metrics[
                        "raw_line_passenger_service_sum"
                    ],
                    "naive_edge_passenger_service": evaluation.system_metrics[
                        "naive_edge_passenger_service"
                    ],
                    "unique_edge_passenger_service": evaluation.system_metrics[
                        "unique_edge_passenger_service"
                    ],
                    "total_route_cost": evaluation.total_route_cost,
                    "unique_edge_cost": evaluation.system_metrics["unique_edge_cost"],
                    "average_line_cost": evaluation.system_metrics["average_line_cost"],
                    "max_line_cost": evaluation.system_metrics["max_line_cost"],
                    "average_segment_cost": evaluation.system_metrics["average_segment_cost"],
                    "max_segment_cost": evaluation.system_metrics["max_segment_cost"],
                    "average_edges_per_line": evaluation.system_metrics[
                        "average_edges_per_line"
                    ],
                    "max_edges_per_line": evaluation.system_metrics["max_edges_per_line"],
                    "system_cost_norm": evaluation.system_cost_norm,
                    "system_service_norm": evaluation.system_service_norm,
                    "shared_stop_ratio": evaluation.shared_stop_ratio,
                    "shared_edge_ratio": evaluation.shared_edge_ratio,
                    "best_system": self._format_system(evaluation.lines),
                }
            )

    def _format_system(self, system: Sequence[Sequence[Node]]) -> str:
        result = " | ".join(
            ";".join(str(stop) for stop in line)
            for line in system
        )
        return result

    def _save_improvement_snapshot(
        self,
        generation: int,
        evaluation: JointEvaluationResult,
    ) -> Optional[str]:
        saved_pdf_path: Optional[str] = None

        if self.snapshot_output_dir is not None:
            os.makedirs(_filesystem_path(self.snapshot_output_dir), exist_ok=True)
            pdf_path = os.path.join(self.snapshot_output_dir, f"generation_{generation:03d}.pdf")
            png_path = os.path.join(self.snapshot_output_dir, f"generation_{generation:03d}.png")
            from busline_ga.visualization.visualize_joint_multiline_ga import save_joint_multiline_map

            save_joint_multiline_map(
                network=self.network,
                lines=evaluation.lines,
                output_pdf=pdf_path,
                output_png=png_path,
            )
            saved_pdf_path = pdf_path

        return saved_pdf_path

    def _record_improvement(
        self,
        generation: int,
        evaluation: JointEvaluationResult,
        pdf_path: Optional[str],
    ) -> None:
        self.improvement_history.append(
            {
                "generation": generation,
                "fitness": evaluation.fitness,
                "base_system_fitness": evaluation.system_metrics["base_system_fitness"],
                "compactness_penalty": evaluation.system_metrics["compactness_penalty"],
                "compactness_penalty_value": evaluation.system_metrics[
                    "compactness_penalty_value"
                ],
                "compactness_total_cost_component": evaluation.system_metrics[
                    "compactness_total_cost_component"
                ],
                "compactness_max_line_excess": evaluation.system_metrics[
                    "compactness_max_line_excess"
                ],
                "compactness_max_segment_excess": evaluation.system_metrics[
                    "compactness_max_segment_excess"
                ],
                "compactness_max_edges_excess": evaluation.system_metrics[
                    "compactness_max_edges_excess"
                ],
                "adjusted_passenger_service": evaluation.adjusted_passenger_service,
                "raw_line_passenger_service_sum": evaluation.system_metrics[
                    "raw_line_passenger_service_sum"
                ],
                "naive_edge_passenger_service": evaluation.system_metrics[
                    "naive_edge_passenger_service"
                ],
                "unique_edge_passenger_service": evaluation.system_metrics[
                    "unique_edge_passenger_service"
                ],
                "total_route_cost": evaluation.total_route_cost,
                "average_line_cost": evaluation.system_metrics["average_line_cost"],
                "max_line_cost": evaluation.system_metrics["max_line_cost"],
                "average_segment_cost": evaluation.system_metrics["average_segment_cost"],
                "max_segment_cost": evaluation.system_metrics["max_segment_cost"],
                "average_edges_per_line": evaluation.system_metrics["average_edges_per_line"],
                "max_edges_per_line": evaluation.system_metrics["max_edges_per_line"],
                "system_cost_norm": evaluation.system_cost_norm,
                "system_service_norm": evaluation.system_service_norm,
                "shared_stop_ratio": evaluation.shared_stop_ratio,
                "shared_edge_ratio": evaluation.shared_edge_ratio,
                "pdf_path": pdf_path or "",
            }
        )

    def _next_generation(
        self,
        scored_population: Sequence[Tuple[SystemIndividual, JointEvaluationResult]],
    ) -> None:
        sorted_population = sorted(scored_population, key=lambda item: item[1].fitness, reverse=True)
        new_population = [
            self._copy_system(system)
            for system, _ in sorted_population[: self.n_elite]
        ]

        while len(new_population) < self.population_size:
            parent_a = self._tournament_select(scored_population)
            parent_b = self._tournament_select(scored_population)
            child = self._crossover(parent_a, parent_b)
            child = self._mutate(child)
            new_population.append(child)

        self.population = new_population

    def run(self) -> Dict[str, Any]:
        self.generate_initial_population()
        best_system: SystemIndividual = []
        best_evaluation: Optional[JointEvaluationResult] = None
        best_score = float("-inf")

        for generation in range(self.generations):
            scored_population = self._evaluate_population()
            generation_best_system, generation_best_evaluation = max(
                scored_population,
                key=lambda item: item[1].fitness,
            )
            self._record_generation(generation, generation_best_evaluation)

            if generation_best_evaluation.fitness > best_score:
                best_system = self._copy_system(generation_best_system)
                best_evaluation = generation_best_evaluation
                best_score = generation_best_evaluation.fitness
                pdf_path: Optional[str] = None

                if self.save_improvement_snapshots:
                    pdf_path = self._save_improvement_snapshot(generation, best_evaluation)

                self._record_improvement(generation, best_evaluation, pdf_path)

            if generation < self.generations - 1:
                self._next_generation(scored_population)

        if best_evaluation is None:
            best_evaluation = self.evaluate_system(best_system, include_details=True)
        else:
            best_evaluation = self.evaluate_system(best_system, include_details=True)

        result = {
            "best_system": best_system,
            "best_evaluation": best_evaluation,
            "lines": best_evaluation.lines,
            "evaluations": best_evaluation.evaluations,
            "system_metrics": best_evaluation.system_metrics,
            "adjusted_line_metrics": best_evaluation.adjusted_line_metrics,
            "pairwise_overlap": best_evaluation.pairwise_overlap,
            "normalization_bounds": self.normalization_bounds,
            "evolution_history": self.evolution_history,
            "improvement_history": self.improvement_history,
        }
        return result
