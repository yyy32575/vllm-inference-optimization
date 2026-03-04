"""
Strategy Comparator for VLLM Inference Optimization.

Compares different optimization strategies including:
- Dynamic vs static batching
- Different max_num_batched_tokens settings
- Prefix cache on/off
- Long/short request routing
"""

import itertools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .benchmark_runner import BenchmarkResult
from .workload_generator import RequestSample

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a single optimization strategy."""

    name: str
    batching_mode: str = "dynamic"  # "dynamic" or "static"
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 128
    prefix_cache_enabled: bool = False
    request_routing_enabled: bool = False
    short_threshold: int = 256
    long_threshold: int = 1024
    gpu_memory_utilization: float = 0.90

    def to_vllm_args(self) -> Dict[str, Any]:
        """Convert to vLLM server launch arguments."""
        args = {
            "max-num-batched-tokens": self.max_num_batched_tokens,
            "max-num-seqs": self.max_num_seqs,
            "gpu-memory-utilization": self.gpu_memory_utilization,
            "enable-prefix-caching": self.prefix_cache_enabled,
        }
        if self.batching_mode == "static":
            args["disable-sliding-window"] = True
        return args

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "batching_mode": self.batching_mode,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "prefix_cache_enabled": self.prefix_cache_enabled,
            "request_routing_enabled": self.request_routing_enabled,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }


@dataclass
class StrategyResult:
    """Result of evaluating a single strategy."""

    config: StrategyConfig
    benchmark_result: BenchmarkResult
    meets_slo: bool = False
    score: float = 0.0  # Composite optimization score


@dataclass
class ComparisonReport:
    """Full comparison report across all strategies."""

    strategy_results: List[StrategyResult]
    best_strategy: Optional[StrategyResult] = None
    baseline_strategy: Optional[StrategyResult] = None
    improvement_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        results = []
        for sr in self.strategy_results:
            results.append({
                "config": sr.config.to_dict(),
                "benchmark": sr.benchmark_result.to_dict(),
                "meets_slo": sr.meets_slo,
                "score": round(sr.score, 4),
            })
        return {
            "strategy_results": results,
            "best_strategy": (
                self.best_strategy.config.name if self.best_strategy else None
            ),
            "improvement_summary": self.improvement_summary,
        }


class StrategyComparator:
    """
    Generates and evaluates different optimization strategy configurations.

    Supports grid search over batching modes, token limits, prefix caching,
    and request routing strategies.
    """

    def __init__(
        self,
        batching_modes: Optional[List[str]] = None,
        max_batched_tokens_options: Optional[List[int]] = None,
        max_num_seqs_options: Optional[List[int]] = None,
        prefix_cache_options: Optional[List[bool]] = None,
        routing_options: Optional[List[bool]] = None,
        p95_latency_limit_ms: float = 3000,
        min_throughput_tps: float = 500,
        weight_latency: float = 0.5,
        weight_throughput: float = 0.5,
    ):
        self.batching_modes = batching_modes or ["dynamic", "static"]
        self.max_batched_tokens_options = max_batched_tokens_options or [
            2048, 4096, 8192, 16384
        ]
        self.max_num_seqs_options = max_num_seqs_options or [64, 128, 256]
        self.prefix_cache_options = prefix_cache_options or [True, False]
        self.routing_options = routing_options or [True, False]
        self.p95_latency_limit_ms = p95_latency_limit_ms
        self.min_throughput_tps = min_throughput_tps
        self.weight_latency = weight_latency
        self.weight_throughput = weight_throughput

    def generate_strategy_configs(self) -> List[StrategyConfig]:
        """Generate all strategy configurations from the parameter grid."""
        configs = []
        combinations = itertools.product(
            self.batching_modes,
            self.max_batched_tokens_options,
            self.max_num_seqs_options,
            self.prefix_cache_options,
            self.routing_options,
        )
        for batch_mode, max_tokens, max_seqs, prefix_cache, routing in combinations:
            name = (
                f"{batch_mode}_bt{max_tokens}_seq{max_seqs}"
                f"_cache{'On' if prefix_cache else 'Off'}"
                f"_route{'On' if routing else 'Off'}"
            )
            configs.append(
                StrategyConfig(
                    name=name,
                    batching_mode=batch_mode,
                    max_num_batched_tokens=max_tokens,
                    max_num_seqs=max_seqs,
                    prefix_cache_enabled=prefix_cache,
                    request_routing_enabled=routing,
                )
            )
        logger.info("Generated %d strategy configurations", len(configs))
        return configs

    def compute_score(
        self, benchmark_result: BenchmarkResult
    ) -> float:
        """
        Compute a composite optimization score.

        Normalizes latency (lower is better) and throughput (higher is better)
        into a weighted score. Higher score = better configuration.
        """
        # Normalize P95 latency: convert to a 0-1 score where lower latency = higher score
        latency_score = max(
            0.0,
            1.0 - (benchmark_result.p95_latency_ms / self.p95_latency_limit_ms),
        )

        # Normalize throughput: convert to a 0-1 score
        throughput_score = min(
            1.0,
            benchmark_result.throughput_tokens_per_sec / self.min_throughput_tps,
        )

        score = (
            self.weight_latency * latency_score
            + self.weight_throughput * throughput_score
        )
        return score

    def evaluate_strategy(
        self,
        config: StrategyConfig,
        benchmark_result: BenchmarkResult,
    ) -> StrategyResult:
        """Evaluate a single strategy against SLO constraints."""
        meets_slo = benchmark_result.meets_slo(
            self.p95_latency_limit_ms, self.min_throughput_tps
        )
        score = self.compute_score(benchmark_result)

        return StrategyResult(
            config=config,
            benchmark_result=benchmark_result,
            meets_slo=meets_slo,
            score=score,
        )

    def compare_strategies(
        self,
        strategy_results: List[StrategyResult],
    ) -> ComparisonReport:
        """
        Compare all evaluated strategies and produce a ranking report.

        Identifies the best strategy among those meeting SLO constraints.
        Falls back to the highest-scoring strategy if none meet SLO.
        """
        if not strategy_results:
            return ComparisonReport(strategy_results=[])

        # Sort by score (descending)
        sorted_results = sorted(
            strategy_results, key=lambda r: r.score, reverse=True
        )

        # Find best among SLO-meeting strategies
        slo_meeting = [r for r in sorted_results if r.meets_slo]
        best = slo_meeting[0] if slo_meeting else sorted_results[0]

        # Assume first result is baseline (or find static/default config)
        baseline = None
        for r in strategy_results:
            if "static" in r.config.name and not r.config.prefix_cache_enabled:
                baseline = r
                break
        if baseline is None:
            baseline = strategy_results[0]

        # Compute improvement summary
        improvement = {}
        if baseline and best and baseline != best:
            b = baseline.benchmark_result
            o = best.benchmark_result
            improvement = {
                "throughput_increase_pct": round(
                    ((o.throughput_tokens_per_sec - b.throughput_tokens_per_sec)
                     / max(b.throughput_tokens_per_sec, 1)) * 100,
                    2,
                ),
                "p95_latency_reduction_pct": round(
                    ((b.p95_latency_ms - o.p95_latency_ms)
                     / max(b.p95_latency_ms, 1)) * 100,
                    2,
                ),
            }

        report = ComparisonReport(
            strategy_results=sorted_results,
            best_strategy=best,
            baseline_strategy=baseline,
            improvement_summary=improvement,
        )

        logger.info(
            "Strategy comparison: %d total, %d meet SLO, best='%s' (score=%.4f)",
            len(strategy_results),
            len(slo_meeting),
            best.config.name,
            best.score,
        )
        return report

    def export_report(
        self, report: ComparisonReport, filepath: str
    ) -> None:
        """Export comparison report to JSON."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Exported comparison report to %s", filepath)


def create_baseline_config() -> StrategyConfig:
    """Create a baseline (unoptimized) strategy configuration."""
    return StrategyConfig(
        name="baseline_static",
        batching_mode="static",
        max_num_batched_tokens=2048,
        max_num_seqs=64,
        prefix_cache_enabled=False,
        request_routing_enabled=False,
    )


def create_optimized_config() -> StrategyConfig:
    """Create a recommended optimized strategy configuration."""
    return StrategyConfig(
        name="optimized_dynamic",
        batching_mode="dynamic",
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        prefix_cache_enabled=True,
        request_routing_enabled=True,
    )
