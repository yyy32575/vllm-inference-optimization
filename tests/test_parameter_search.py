"""Tests for SLO-constrained parameter search module."""

import json
import os
import tempfile

import pytest

from src.benchmark_runner import BenchmarkResult
from src.parameter_search import (
    ParameterSearch,
    SLOConstraints,
    SearchResult,
    SearchSpace,
)
from src.strategy_comparator import StrategyConfig


def _make_benchmark_result(
    name: str = "test",
    p95_latency_ms: float = 2000.0,
    p99_latency_ms: float = 4000.0,
    throughput_tps: float = 600.0,
    ttft_ms: float = 300.0,
) -> BenchmarkResult:
    """Helper to create a benchmark result for testing."""
    return BenchmarkResult(
        config_name=name,
        total_requests=100,
        successful_requests=95,
        failed_requests=5,
        total_duration_seconds=10.0,
        avg_latency_ms=1000.0,
        p50_latency_ms=800.0,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p99_latency_ms,
        avg_ttft_ms=ttft_ms,
        p95_ttft_ms=ttft_ms * 1.5,
        throughput_requests_per_sec=9.5,
        throughput_tokens_per_sec=throughput_tps,
        avg_tokens_per_request=63.0,
        avg_prefill_latency_ms=300.0,
        avg_decode_latency_ms=700.0,
        prefill_ratio=0.3,
    )


class TestSLOConstraints:
    """Tests for SLOConstraints."""

    def test_default_constraints(self):
        slo = SLOConstraints()
        assert slo.p95_latency_ms == 3000.0
        assert slo.min_throughput_tokens_per_sec == 500.0

    def test_satisfied(self):
        slo = SLOConstraints(
            p95_latency_ms=3000,
            p99_latency_ms=10000,
            min_throughput_tokens_per_sec=500,
            max_ttft_ms=1000,
        )
        result = _make_benchmark_result(
            p95_latency_ms=2000,
            p99_latency_ms=4000,
            throughput_tps=600,
            ttft_ms=300,
        )
        assert slo.is_satisfied(result) is True

    def test_violated_latency(self):
        slo = SLOConstraints(p95_latency_ms=1000)
        result = _make_benchmark_result(p95_latency_ms=2000)
        assert slo.is_satisfied(result) is False

    def test_violated_throughput(self):
        slo = SLOConstraints(min_throughput_tokens_per_sec=1000)
        result = _make_benchmark_result(throughput_tps=500)
        assert slo.is_satisfied(result) is False

    def test_violated_p99(self):
        slo = SLOConstraints(p99_latency_ms=3000)
        result = _make_benchmark_result(p99_latency_ms=5000)
        assert slo.is_satisfied(result) is False

    def test_violated_ttft(self):
        slo = SLOConstraints(max_ttft_ms=200)
        result = _make_benchmark_result(ttft_ms=300)
        assert slo.is_satisfied(result) is False

    def test_violation_distance_zero(self):
        slo = SLOConstraints(
            p95_latency_ms=3000,
            min_throughput_tokens_per_sec=500,
        )
        result = _make_benchmark_result(
            p95_latency_ms=2000, throughput_tps=600
        )
        assert slo.violation_distance(result) == 0.0

    def test_violation_distance_positive(self):
        slo = SLOConstraints(
            p95_latency_ms=1000,
            min_throughput_tokens_per_sec=1000,
        )
        result = _make_benchmark_result(
            p95_latency_ms=2000, throughput_tps=500
        )
        distance = slo.violation_distance(result)
        assert distance > 0.0


class TestSearchSpace:
    """Tests for SearchSpace."""

    def test_default_space(self):
        space = SearchSpace()
        assert len(space.max_num_batched_tokens) > 0
        assert len(space.batching_mode) > 0

    def test_total_combinations(self):
        space = SearchSpace(
            max_num_batched_tokens=[4096, 8192],
            max_num_seqs=[128],
            gpu_memory_utilization=[0.90],
            prefix_cache=[True, False],
            batching_mode=["dynamic"],
        )
        assert space.total_combinations() == 4  # 2 × 1 × 1 × 2 × 1


class TestParameterSearch:
    """Tests for ParameterSearch."""

    def test_generate_configs(self):
        space = SearchSpace(
            max_num_batched_tokens=[4096],
            max_num_seqs=[128],
            gpu_memory_utilization=[0.90],
            prefix_cache=[True, False],
            batching_mode=["dynamic"],
        )
        search = ParameterSearch(search_space=space)
        configs = search.generate_configs()
        assert len(configs) == 2

    def test_compute_score_good(self):
        search = ParameterSearch(
            slo_constraints=SLOConstraints(
                p95_latency_ms=3000,
                min_throughput_tokens_per_sec=500,
            )
        )
        good_result = _make_benchmark_result(
            p95_latency_ms=1000, throughput_tps=800
        )
        score = search.compute_score(good_result)
        assert score > 0.5

    def test_compute_score_bad(self):
        search = ParameterSearch(
            slo_constraints=SLOConstraints(
                p95_latency_ms=3000,
                min_throughput_tokens_per_sec=500,
            )
        )
        bad_result = _make_benchmark_result(
            p95_latency_ms=5000, throughput_tps=200
        )
        score = search.compute_score(bad_result)
        assert score < 0.5

    def test_compute_score_slo_bonus(self):
        search = ParameterSearch(
            slo_constraints=SLOConstraints(
                p95_latency_ms=3000,
                p99_latency_ms=10000,
                min_throughput_tokens_per_sec=500,
                max_ttft_ms=1000,
            )
        )
        # Meeting SLO should get a bonus
        good = _make_benchmark_result(
            p95_latency_ms=2000, throughput_tps=600, ttft_ms=300
        )
        bad = _make_benchmark_result(
            p95_latency_ms=4000, throughput_tps=600, ttft_ms=300
        )
        good_score = search.compute_score(good)
        bad_score = search.compute_score(bad)
        assert good_score > bad_score

    def test_evaluate_results(self):
        search = ParameterSearch(
            slo_constraints=SLOConstraints(
                p95_latency_ms=3000,
                p99_latency_ms=10000,
                min_throughput_tokens_per_sec=500,
                max_ttft_ms=1000,
            )
        )

        results = [
            (
                StrategyConfig(name="config_a"),
                _make_benchmark_result(
                    name="config_a",
                    p95_latency_ms=1500,
                    throughput_tps=700,
                    ttft_ms=200,
                ),
            ),
            (
                StrategyConfig(name="config_b"),
                _make_benchmark_result(
                    name="config_b",
                    p95_latency_ms=4000,
                    throughput_tps=300,
                    ttft_ms=800,
                ),
            ),
        ]

        search_result = search.evaluate_results(results)
        assert isinstance(search_result, SearchResult)
        assert search_result.configs_evaluated == 2
        assert search_result.best_config is not None
        assert search_result.best_config.name == "config_a"
        assert search_result.best_meets_slo is True
        assert search_result.configs_meeting_slo >= 1

    def test_evaluate_no_slo_meeting(self):
        search = ParameterSearch(
            slo_constraints=SLOConstraints(
                p95_latency_ms=500,  # Very strict
                min_throughput_tokens_per_sec=2000,  # Very strict
            )
        )

        results = [
            (
                StrategyConfig(name="config_a"),
                _make_benchmark_result(
                    p95_latency_ms=2000, throughput_tps=600
                ),
            ),
        ]

        search_result = search.evaluate_results(results)
        assert search_result.configs_meeting_slo == 0
        # Should still pick the best available
        assert search_result.best_config is not None

    def test_pareto_frontier(self):
        search = ParameterSearch()

        results = [
            (
                StrategyConfig(name="fast_low_throughput"),
                _make_benchmark_result(
                    name="fast_low_throughput",
                    p95_latency_ms=500,
                    throughput_tps=300,
                ),
            ),
            (
                StrategyConfig(name="slow_high_throughput"),
                _make_benchmark_result(
                    name="slow_high_throughput",
                    p95_latency_ms=3000,
                    throughput_tps=900,
                ),
            ),
            (
                StrategyConfig(name="dominated"),
                _make_benchmark_result(
                    name="dominated",
                    p95_latency_ms=3000,
                    throughput_tps=300,
                ),
            ),
        ]

        search_result = search.evaluate_results(results)
        pareto_names = [r.config.name for r in search_result.pareto_frontier]
        # "dominated" should not be on the Pareto frontier
        assert "dominated" not in pareto_names
        assert "fast_low_throughput" in pareto_names
        assert "slow_high_throughput" in pareto_names

    def test_evaluate_empty(self):
        search = ParameterSearch()
        search_result = search.evaluate_results([])
        assert search_result.best_config is None
        assert search_result.configs_evaluated == 0

    def test_export_results(self):
        search = ParameterSearch()
        results = [
            (
                StrategyConfig(name="test"),
                _make_benchmark_result(name="test"),
            ),
        ]
        search_result = search.evaluate_results(results)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            filepath = f.name

        try:
            search.export_results(search_result, filepath)
            with open(filepath, "r") as f:
                data = json.load(f)
            assert "best_config" in data
            assert "pareto_frontier" in data
        finally:
            os.unlink(filepath)

    def test_search_result_to_dict(self):
        search = ParameterSearch()
        results = [
            (
                StrategyConfig(name="test"),
                _make_benchmark_result(name="test"),
            ),
        ]
        search_result = search.evaluate_results(results)
        d = search_result.to_dict()
        assert isinstance(d, dict)
        assert "best_config" in d
        assert "configs_evaluated" in d
