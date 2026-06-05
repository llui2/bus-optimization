from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from busline_ga.core.genetic_optimizer import GeneticOptimizer
from busline_ga.core.multi_line_objective import MultiLineObjectiveFunction
from busline_ga.core.network_model import NetworkModel, Node
from busline_ga.core.normalization_utils import prepare_structural_normalization
from busline_ga.core.multi_line_metrics import (
    compute_adjusted_line_metrics,
    compute_multiline_system_metrics,
    compute_pairwise_line_overlap,
)
from busline_ga.core.objective_function import EvaluationResult, NormalizationBounds, ObjectiveFunction
from busline_ga.visualization.multiline_snapshot_plotter import save_multiline_improvement_snapshot


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

        init_ratio_service: float = 0.00,
        init_ratio_demand: float = 0.00,
        init_ratio_spatial: float = 0.00,
        init_ratio_hybrid: float = 0.00,
        init_ratio_random: float = 1.00,
        inject_best_service_candidate: bool = False,

        #init_ratio_service: float = 0.30,
        #init_ratio_demand: float = 0.25,
        #init_ratio_spatial: float = 0.15,
        #init_ratio_hybrid: float = 0.15,
        #init_ratio_random: float = 0.15,
        init_top_k: int = 8,
        #inject_best_service_candidate: bool = True,
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
        save_improvement_snapshots: bool = False,
        snapshot_output_dir: Optional[str] = None,
        save_evolution_history: bool = False,
        evolution_output_dir: Optional[str] = None,
        od_case: Optional[str] = None,
        map_case: Optional[str] = None,
        od_min_to_plot: float = 10.0,
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
        self.save_improvement_snapshots = save_improvement_snapshots
        self.snapshot_output_dir = snapshot_output_dir
        self.save_evolution_history = save_evolution_history
        self.evolution_output_dir = evolution_output_dir
        self.od_case = od_case
        self.map_case = map_case
        self.od_min_to_plot = od_min_to_plot
        self.normalization_bounds = self._prepare_normalization_bounds()
        self.lines: List[List[Node]] = []
        self.evaluations: List[EvaluationResult] = []
        self.histories: List[Dict[str, List[float]]] = []
        self.improvement_snapshots: List[Dict[str, Any]] = []
        self.evolution_history: List[Dict[str, Any]] = []
        self.system_evolution_history: List[Dict[str, Any]] = []

    def _get_service_norm(self, evaluation: EvaluationResult) -> float:
        service_norm = getattr(evaluation, "service_norm", None)

        if service_norm is None:
            service_norm = getattr(evaluation, "passenger_service_norm", 0.0)

        return float(service_norm)

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
        line_snapshot_dir: Optional[str] = None
        improvement_callback = None
        generation_callback = None

        objective = self._build_multi_line_objective(fixed_lines)

        optimizer = self._build_genetic_optimizer(objective, line_index)

        if self.save_improvement_snapshots and self.snapshot_output_dir is not None:
            line_snapshot_dir = os.path.join(self.snapshot_output_dir, f"line_{line_index + 1:02d}")
            os.makedirs(line_snapshot_dir, exist_ok=True)

            def improvement_callback(
                generation: int,
                best_individual: List[Node],
                best_evaluation: EvaluationResult,
            ) -> None:
                pdf_path = os.path.join(line_snapshot_dir, f"generation_{generation:03d}.pdf")
                saved_pdf_path = save_multiline_improvement_snapshot(
                    network=self.network,
                    fixed_lines=fixed_lines,
                    candidate_line=best_individual,
                    fixed_evaluations=self.evaluations,
                    candidate_evaluation=best_evaluation,
                    output_path=pdf_path,
                    line_index=line_index + 1,
                    generation=generation,
                    title_context={
                        "od_case": self.od_case or "",
                        "map_case": self.map_case or "",
                        "shared_stop_penalty": self.shared_stop_penalty,
                    },
                    od_min_to_plot=self.od_min_to_plot,
                )
                self.improvement_snapshots.append(
                    {
                        "line_index": line_index + 1,
                        "generation": generation,
                        "fitness": best_evaluation.fitness,
                        "cost": best_evaluation.cost,
                        "passenger_service": best_evaluation.passenger_service,
                        "cost_norm": best_evaluation.cost_norm,
                        "service_norm": best_evaluation.service_norm,
                        "pdf_path": saved_pdf_path,
                    }
                )
                print(
                    f"[MULTILINE SNAPSHOT SAVED] line={line_index + 1:02d} "
                    f"gen={generation:03d} fitness={best_evaluation.fitness:.12f} "
                    f"pdf={saved_pdf_path}"
                )

        if self.save_evolution_history:
            def generation_callback(
                generation: int,
                generation_best: List[Node],
                best_evaluation: EvaluationResult,
            ) -> None:
                self.evolution_history.append(
                    {
                        "line_index": line_index + 1,
                        "generation": generation,
                        "best_individual": ";".join(str(stop) for stop in generation_best),
                        "fitness": best_evaluation.fitness,
                        "cost": best_evaluation.cost,
                        "cost_norm": best_evaluation.cost_norm,
                        "passenger_service": best_evaluation.passenger_service,
                        "service_norm": self._get_service_norm(best_evaluation),
                    }
                )

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
            improvement_callback=improvement_callback,
            generation_callback=generation_callback,
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

            if self.save_evolution_history:
                partial_system_metrics = compute_multiline_system_metrics(
                    lines=self.lines,
                    evaluations=self.evaluations,
                    network=self.network,
                    normalization_bounds=self.normalization_bounds,
                    lambda_=self.lambda_,
                )
                self.system_evolution_history.append(
                    {
                        "line_index_added": line_index + 1,
                        "n_lines_so_far": len(self.lines),
                        "system_fitness": partial_system_metrics["system_fitness"],
                        "adjusted_passenger_service": partial_system_metrics["adjusted_passenger_service"],
                        "naive_passenger_service": partial_system_metrics["naive_passenger_service"],
                        "total_route_cost": partial_system_metrics["total_route_cost"],
                        "unique_edge_cost": partial_system_metrics["unique_edge_cost"],
                        "system_cost_norm": partial_system_metrics["system_cost_norm"],
                        "system_service_norm": partial_system_metrics["system_service_norm"],
                        "shared_stop_ratio": partial_system_metrics["shared_stop_ratio"],
                        "shared_edge_ratio": partial_system_metrics["shared_edge_ratio"],
                    }
                )

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
        adjusted_line_metrics = compute_adjusted_line_metrics(
            lines=self.lines,
            evaluations=self.evaluations,
            network=self.network,
            normalization_bounds=self.normalization_bounds,
            lambda_=self.lambda_,
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
            "adjusted_line_metrics": adjusted_line_metrics,
            "improvement_snapshots": self.improvement_snapshots,
            "evolution_history": self.evolution_history,
            "system_evolution_history": self.system_evolution_history,
        }
        return result
