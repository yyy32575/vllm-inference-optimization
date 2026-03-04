"""
SLO-Constrained Parameter Search for VLLM Inference Optimization.

Implements a dual-objective optimization process that searches for the
best vLLM configuration parameters under SLO constraints on P95 latency
and throughput (tokens/s).
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .benchmark_runner import BenchmarkResult
from .strategy_comparator import StrategyConfig, StrategyResult

logger = logging.getLogger(__name__)


@dataclass
class SLOConstraints:
    """Service Level Objective constraints for parameter search."""

    p95_latency_ms: float = 3000.0
    p99_latency_ms: float = 10000.0
    min_throughput_tokens_per_sec: float = 500.0
    max_ttft_ms: float = 1000.0
    max_gpu_memory_utilization: float = 0.95

    def is_satisfied(self, result: BenchmarkResult) -> bool:
        """Check if a benchmark result satisfies all SLO constraints."""
        if result.p95_latency_ms > self.p95_latency_ms:
            return False
        if result.p99_latency_ms > self.p99_latency_ms:
            return False
        if result.throughput_tokens_per_sec < self.min_throughput_tokens_per_sec:
            return False
        if result.p95_ttft_ms > self.max_ttft_ms:
            return False
        return True

    def violation_distance(self, result: BenchmarkResult) -> float:
        """
        Compute how far a result is from satisfying constraints.

        Returns 0.0 if all constraints are met, positive value otherwise.
        Higher values mean further from satisfaction.
        """
        violations = []
        if result.p95_latency_ms > self.p95_latency_ms:
            violations.append(
                (result.p95_latency_ms - self.p95_latency_ms) / self.p95_latency_ms
            )
        if result.throughput_tokens_per_sec < self.min_throughput_tokens_per_sec:
            violations.append(
                (self.min_throughput_tokens_per_sec - result.throughput_tokens_per_sec)
                / self.min_throughput_tokens_per_sec
            )
        return sum(violations)


@dataclass
class SearchSpace:
    """Defines the parameter search space."""

    max_num_batched_tokens: List[int] = field(
        default_factory=lambda: [2048, 4096, 8192, 16384]
    )
    max_num_seqs: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )
    gpu_memory_utilization: List[float] = field(
        default_factory=lambda: [0.85, 0.90, 0.95]
    )
    prefix_cache: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    batching_mode: List[str] = field(
        default_factory=lambda: ["dynamic", "static"]
    )

    def total_combinations(self) -> int:
        """Total number of parameter combinations."""
        return (
            len(self.max_num_batched_tokens)
            * len(self.max_num_seqs)
            * len(self.gpu_memory_utilization)
            * len(self.prefix_cache)
            * len(self.batching_mode)
        )


@dataclass
class SearchResult:
    """Result of a parameter search run."""

    best_config: Optional[StrategyConfig]
    best_score: float
    best_meets_slo: bool
    all_results: List[StrategyResult]
    search_duration_seconds: float
    configs_evaluated: int
    configs_meeting_slo: int
    pareto_frontier: List[StrategyResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_config": (
                self.best_config.to_dict() if self.best_config else None
            ),
            "best_score": round(self.best_score, 4),
            "best_meets_slo": self.best_meets_slo,
            "search_duration_seconds": round(self.search_duration_seconds, 2),
            "configs_evaluated": self.configs_evaluated,
            "configs_meeting_slo": self.configs_meeting_slo,
            "pareto_frontier": [
                {
                    "config": r.config.name,
                    "p95_latency_ms": round(r.benchmark_result.p95_latency_ms, 2),
                    "throughput_tps": round(
                        r.benchmark_result.throughput_tokens_per_sec, 2
                    ),
                    "score": round(r.score, 4),
                }
                for r in self.pareto_frontier
            ],
        }


class ParameterSearch:
    """
    SLO-constrained parameter search engine.

    Performs grid search or guided search over vLLM configuration parameters,
    optimizing for both P95 latency and throughput under SLO constraints.
    """

    def __init__(
        self,
        slo_constraints: Optional[SLOConstraints] = None,
        search_space: Optional[SearchSpace] = None,
        weight_latency: float = 0.5,
        weight_throughput: float = 0.5,
    ):
        self.slo = slo_constraints or SLOConstraints()
        self.search_space = search_space or SearchSpace()
        self.weight_latency = weight_latency
        self.weight_throughput = weight_throughput

    def compute_score(
        self, result: BenchmarkResult
    ) -> float:
        """
        Compute dual-objective optimization score.

        Combines normalized latency (lower=better) and throughput (higher=better)
        into a single score using configured weights.
        """
        # Latency score: 1.0 when latency=0, 0.0 when latency=slo_limit
        latency_score = max(
            0.0,
            1.0 - (result.p95_latency_ms / self.slo.p95_latency_ms),
        )

        # Throughput score: 1.0 when throughput >= target
        throughput_score = min(
            1.0,
            result.throughput_tokens_per_sec / self.slo.min_throughput_tokens_per_sec,
        )

        # Bonus for meeting all SLO constraints
        slo_bonus = 0.2 if self.slo.is_satisfied(result) else 0.0

        score = (
            self.weight_latency * latency_score
            + self.weight_throughput * throughput_score
            + slo_bonus
        )
        return score

    def generate_configs(self) -> List[StrategyConfig]:
        """Generate all configurations from the search space."""
        configs = []
        idx = 0
        for bt in self.search_space.max_num_batched_tokens:
            for seqs in self.search_space.max_num_seqs:
                for gpu_util in self.search_space.gpu_memory_utilization:
                    for cache in self.search_space.prefix_cache:
                        for mode in self.search_space.batching_mode:
                            name = (
                                f"search_{idx:03d}_{mode}_bt{bt}_seq{seqs}"
                                f"_gpu{int(gpu_util*100)}"
                                f"_cache{'On' if cache else 'Off'}"
                            )
                            configs.append(
                                StrategyConfig(
                                    name=name,
                                    batching_mode=mode,
                                    max_num_batched_tokens=bt,
                                    max_num_seqs=seqs,
                                    prefix_cache_enabled=cache,
                                    gpu_memory_utilization=gpu_util,
                                )
                            )
                            idx += 1
        return configs

    def evaluate_results(
        self,
        results: List[Tuple[StrategyConfig, BenchmarkResult]],
    ) -> SearchResult:
        """
        Evaluate all benchmark results and find the optimal configuration.

        Computes scores, identifies Pareto frontier, and selects the
        best configuration meeting SLO constraints.
        """
        start = time.perf_counter()
        strategy_results = []

        for config, benchmark in results:
            score = self.compute_score(benchmark)
            meets_slo = self.slo.is_satisfied(benchmark)
            strategy_results.append(
                StrategyResult(
                    config=config,
                    benchmark_result=benchmark,
                    meets_slo=meets_slo,
                    score=score,
                )
            )

        # Sort by score descending
        strategy_results.sort(key=lambda r: r.score, reverse=True)

        # Find Pareto frontier (non-dominated solutions)
        pareto = self._compute_pareto_frontier(strategy_results)

        # Select best: prefer SLO-meeting configs, then highest score
        slo_meeting = [r for r in strategy_results if r.meets_slo]
        if slo_meeting:
            best = slo_meeting[0]
        elif strategy_results:
            best = strategy_results[0]
        else:
            best = None

        duration = time.perf_counter() - start
        return SearchResult(
            best_config=best.config if best else None,
            best_score=best.score if best else 0.0,
            best_meets_slo=best.meets_slo if best else False,
            all_results=strategy_results,
            search_duration_seconds=duration,
            configs_evaluated=len(results),
            configs_meeting_slo=len(slo_meeting),
            pareto_frontier=pareto,
        )

    def _compute_pareto_frontier(
        self, results: List[StrategyResult]
    ) -> List[StrategyResult]:
        """
        Compute the Pareto frontier for the dual objectives:
        - Minimize P95 latency
        - Maximize throughput (tokens/s)

        A result is Pareto-optimal if no other result is better in both objectives.
        """
        if not results:
            return []

        pareto = []
        for candidate in results:
            is_dominated = False
            c_lat = candidate.benchmark_result.p95_latency_ms
            c_thr = candidate.benchmark_result.throughput_tokens_per_sec

            for other in results:
                if other is candidate:
                    continue
                o_lat = other.benchmark_result.p95_latency_ms
                o_thr = other.benchmark_result.throughput_tokens_per_sec

                # other dominates candidate if it's better in both objectives
                if o_lat <= c_lat and o_thr >= c_thr and (
                    o_lat < c_lat or o_thr > c_thr
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append(candidate)

        # Sort by latency ascending
        pareto.sort(key=lambda r: r.benchmark_result.p95_latency_ms)
        return pareto

    def export_results(
        self, search_result: SearchResult, filepath: str
    ) -> None:
        """Export search results to JSON."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(search_result.to_dict(), f, indent=2)
        logger.info("Exported search results to %s", filepath)
