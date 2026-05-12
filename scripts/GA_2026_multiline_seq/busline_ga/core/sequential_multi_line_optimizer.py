from __future__ import annotations

from typing import Any, Dict, List, Optional

from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.multi_line_objective import MultiLineObjectiveFunction
from busline_ga.core.network_model import NetworkModel, Node
from busline_ga.core.normalization_utils import prepare_structural_normalization
from busline_ga.core.multi_line_metrics import (
    compute_multiline_system_metrics,
    compute_pairwise_line_overlap,
)
from busline_ga.core.objective_function import EvaluationResult, NormalizationBounds, ObjectiveFunction


class SequentialMultiLineOptimizer:
    def __init__(
        self,
        network: NetworkModel,
        line_length: int,
        population_size: int,
        n_lines: int,
        lambda_: float,
        generations: int,
        seed: int = 42,
        nearest_neighbors_k: int = 8,
        init_ratio_service: float = 0.30,
        init_ratio_demand: float = 0.25,
        init_ratio_spatial: float = 0.15,
        init_ratio_hybrid: float = 0.15,
        init_ratio_random: float = 0.15,
        init_top_k: int = 8,
        inject_best_service_candidate: bool = True,
        n_elite: int = 2,
        tournament_size: int = 3,
        mutation_prob: float = 0.30,
        weight_adjacent: float = 0.20,
        weight_reverse: float = 0.15,
        weight_neighbor: float = 0.25,
        weight_service_replace: float = 0.40,
        max_reverse_length: int = 4,
        acceptance_tolerance: float = 0.00,
        normalization_warnings: bool = False,
        shared_stop_penalty: float = 0.0,
    ) -> None:
        self.network = network
        self.line_length = line_length
        self.population_size = population_size
        self.n_lines = n_lines
        self.lambda_ = lambda_
        self.generations = generations
        self.seed = seed
        self.nearest_neighbors_k = nearest_neighbors_k
        self.init_ratio_service = init_ratio_service
        self.init_ratio_demand = init_ratio_demand
        self.init_ratio_spatial = init_ratio_spatial
        self.init_ratio_hybrid = init_ratio_hybrid
        self.init_ratio_random = init_ratio_random
        self.init_top_k = init_top_k
        self.inject_best_service_candidate = inject_best_service_candidate
        self.n_elite = n_elite
        self.tournament_size = tournament_size
        self.mutation_prob = mutation_prob
        self.weight_adjacent = weight_adjacent
        self.weight_reverse = weight_reverse
        self.weight_neighbor = weight_neighbor
        self.weight_service_replace = weight_service_replace
        self.max_reverse_length = max_reverse_length
        self.acceptance_tolerance = acceptance_tolerance
        self.normalization_warnings = normalization_warnings
        self.shared_stop_penalty = shared_stop_penalty
        self.normalization_bounds = self._prepare_normalization_bounds()
        self.lines: List[List[Node]] = []
        self.evaluations: List[EvaluationResult] = []
        self.histories: List[Dict[str, List[float]]] = []

    def _prepare_normalization_bounds(self) -> NormalizationBounds:
        base_objective = ObjectiveFunction(
            network=self.network,
            normalization_warnings=self.normalization_warnings,
        )
        bounds = prepare_structural_normalization(
            objective=base_objective,
            line_length=self.line_length,
        )
        result = bounds
        return result

    def _build_single_line_objective(self) -> ObjectiveFunction:
        objective = ObjectiveFunction(
            network=self.network,
            normalization_warnings=self.normalization_warnings,
        )
        objective.precompute_edge_costs()
        objective.precompute_edge_service()
        objective.set_normalization_bounds(self.normalization_bounds)
        result = objective
        return result

    def _build_multi_line_objective(
        self,
        fixed_lines: List[List[Node]],
    ) -> MultiLineObjectiveFunction:
        objective = MultiLineObjectiveFunction(
            network=self.network,
            fixed_lines=fixed_lines,
            normalization_bounds=self.normalization_bounds,
            normalization_warnings=self.normalization_warnings,
            shared_stop_penalty=self.shared_stop_penalty,
        )
        result = objective
        return result

    def _build_genetic_optimizer(
        self,
        objective_function: ObjectiveFunction,
        line_index: int,
    ) -> GeneticOptimizer:
        optimizer = GeneticOptimizer(
            network=self.network,
            objective_function=objective_function,
            line_length=self.line_length,
            population_size=self.population_size,
            seed=self.seed + line_index,
            nearest_neighbors_k=self.nearest_neighbors_k,
            init_ratio_service=self.init_ratio_service,
            init_ratio_demand=self.init_ratio_demand,
            init_ratio_spatial=self.init_ratio_spatial,
            init_ratio_hybrid=self.init_ratio_hybrid,
            init_ratio_random=self.init_ratio_random,
            init_top_k=self.init_top_k,
            inject_best_service_candidate=self.inject_best_service_candidate,
        )
        result = optimizer
        return result

    def _optimize_line(
        self,
        line_index: int,
        fixed_lines: List[List[Node]],
    ) -> Dict[str, Any]:
        objective: ObjectiveFunction

        if fixed_lines:
            objective = self._build_multi_line_objective(fixed_lines)
        else:
            objective = self._build_single_line_objective()

        optimizer = self._build_genetic_optimizer(objective, line_index)
        best_individual, best_score, history = optimizer.run(
            lambda_=self.lambda_,
            generations=self.generations,
            n_elite=self.n_elite,
            tournament_size=self.tournament_size,
            mutation_prob=self.mutation_prob,
            weight_adjacent=self.weight_adjacent,
            weight_reverse=self.weight_reverse,
            weight_neighbor=self.weight_neighbor,
            weight_service_replace=self.weight_service_replace,
            max_reverse_length=self.max_reverse_length,
            acceptance_tolerance=self.acceptance_tolerance,
            return_history=True,
            save_improvement_snapshots=False,
        )
        evaluation = objective.evaluate(best_individual, self.lambda_)
        line_result = {
            "line_index": line_index,
            "line": best_individual,
            "evaluation": evaluation,
            "fitness": best_score,
            "history": history,
        }
        return line_result

    def run(self) -> Dict[str, Any]:
        fixed_lines: List[List[Node]] = []
        line_results: List[Dict[str, Any]] = []

        for line_index in range(self.n_lines):
            line_result = self._optimize_line(line_index, fixed_lines)
            best_line = list(line_result["line"])
            evaluation = line_result["evaluation"]
            fixed_lines.append(best_line)
            line_results.append(line_result)
            self.lines.append(best_line)
            self.evaluations.append(evaluation)
            self.histories.append(line_result["history"])

        system_metrics = compute_multiline_system_metrics(
            lines=self.lines,
            evaluations=self.evaluations,
            network=self.network,
            normalization_bounds=self.normalization_bounds,
            lambda_=self.lambda_,
        )
        pairwise_overlap = compute_pairwise_line_overlap(
            lines=self.lines,
            evaluations=self.evaluations,
        )

        result = {
            "lines": self.lines,
            "evaluations": self.evaluations,
            "fitnesses": [evaluation.fitness for evaluation in self.evaluations],
            "costs": [evaluation.cost for evaluation in self.evaluations],
            "services": [evaluation.passenger_service for evaluation in self.evaluations],
            "line_results": line_results,
            "normalization_bounds": self.normalization_bounds,
            "system_metrics": system_metrics,
            "pairwise_overlap": pairwise_overlap,
        }
        return result
