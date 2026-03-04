"""Tests for strategy comparator module."""

import json
import os
import tempfile

import pytest

from src.benchmark_runner import BenchmarkResult
from src.strategy_comparator import (
    ComparisonReport,
    StrategyComparator,
    StrategyConfig,
    StrategyResult,
    create_baseline_config,
    create_optimized_config,
)


def _make_benchmark_result(
    name: str = "test",
    p95_latency_ms: float = 2000.0,
    p99_latency_ms: float = 4000.0,
    throughput_tps: float = 600.0,
    avg_latency_ms: float = 1000.0,
    prefill_ms: float = 300.0,
    decode_ms: float = 700.0,
) -> BenchmarkResult:
    """Helper to create a benchmark result for testing."""
    return BenchmarkResult(
        config_name=name,
        total_requests=100,
        successful_requests=95,
        failed_requests=5,
        total_duration_seconds=10.0,
        avg_latency_ms=avg_latency_ms,
        p50_latency_ms=avg_latency_ms * 0.8,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p99_latency_ms,
        avg_ttft_ms=prefill_ms,
        p95_ttft_ms=prefill_ms * 1.5,
        throughput_requests_per_sec=9.5,
        throughput_tokens_per_sec=throughput_tps,
        avg_tokens_per_request=63.0,
        avg_prefill_latency_ms=prefill_ms,
        avg_decode_latency_ms=decode_ms,
        prefill_ratio=prefill_ms / (prefill_ms + decode_ms),
    )


class TestStrategyConfig:
    """Tests for StrategyConfig."""

    def test_default_config(self):
        config = StrategyConfig(name="test")
        assert config.batching_mode == "dynamic"
        assert config.max_num_batched_tokens == 4096
        assert config.max_num_seqs == 128
        assert config.prefix_cache_enabled is False

    def test_to_vllm_args(self):
        config = StrategyConfig(
            name="test",
            max_num_batched_tokens=8192,
            max_num_seqs=256,
            prefix_cache_enabled=True,
        )
        args = config.to_vllm_args()
        assert args["max-num-batched-tokens"] == 8192
        assert args["max-num-seqs"] == 256
        assert args["enable-prefix-caching"] is True

    def test_to_dict(self):
        config = StrategyConfig(name="test", batching_mode="static")
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["batching_mode"] == "static"

    def test_baseline_config(self):
        config = create_baseline_config()
        assert config.batching_mode == "static"
        assert config.prefix_cache_enabled is False

    def test_optimized_config(self):
        config = create_optimized_config()
        assert config.batching_mode == "dynamic"
        assert config.prefix_cache_enabled is True
        assert config.request_routing_enabled is True


class TestBenchmarkResultSLO:
    """Tests for BenchmarkResult SLO checking."""

    def test_meets_slo(self):
        result = _make_benchmark_result(
            p95_latency_ms=2000, throughput_tps=600
        )
        assert result.meets_slo(3000, 500) is True

    def test_fails_slo_latency(self):
        result = _make_benchmark_result(
            p95_latency_ms=5000, throughput_tps=600
        )
        assert result.meets_slo(3000, 500) is False

    def test_fails_slo_throughput(self):
        result = _make_benchmark_result(
            p95_latency_ms=2000, throughput_tps=300
        )
        assert result.meets_slo(3000, 500) is False


class TestStrategyComparator:
    """Tests for StrategyComparator."""

    def test_generate_configs(self):
        comparator = StrategyComparator(
            batching_modes=["dynamic", "static"],
            max_batched_tokens_options=[4096, 8192],
            max_num_seqs_options=[128],
            prefix_cache_options=[True, False],
            routing_options=[False],
        )
        configs = comparator.generate_strategy_configs()
        # 2 modes × 2 token options × 1 seq option × 2 cache × 1 routing = 8
        assert len(configs) == 8

    def test_config_names_unique(self):
        comparator = StrategyComparator(
            batching_modes=["dynamic"],
            max_batched_tokens_options=[4096],
            max_num_seqs_options=[128],
            prefix_cache_options=[True, False],
            routing_options=[True, False],
        )
        configs = comparator.generate_strategy_configs()
        names = [c.name for c in configs]
        assert len(set(names)) == len(names)

    def test_compute_score(self):
        comparator = StrategyComparator(
            p95_latency_limit_ms=3000,
            min_throughput_tps=500,
            weight_latency=0.5,
            weight_throughput=0.5,
        )
        # Good result: low latency, high throughput
        good = _make_benchmark_result(p95_latency_ms=1500, throughput_tps=600)
        # Bad result: high latency, low throughput
        bad = _make_benchmark_result(p95_latency_ms=5000, throughput_tps=200)

        good_score = comparator.compute_score(good)
        bad_score = comparator.compute_score(bad)
        assert good_score > bad_score

    def test_evaluate_strategy_meets_slo(self):
        comparator = StrategyComparator(
            p95_latency_limit_ms=3000, min_throughput_tps=500
        )
        config = StrategyConfig(name="good")
        result = _make_benchmark_result(p95_latency_ms=2000, throughput_tps=600)
        sr = comparator.evaluate_strategy(config, result)
        assert sr.meets_slo is True
        assert sr.score > 0

    def test_evaluate_strategy_fails_slo(self):
        comparator = StrategyComparator(
            p95_latency_limit_ms=3000, min_throughput_tps=500
        )
        config = StrategyConfig(name="bad")
        result = _make_benchmark_result(p95_latency_ms=5000, throughput_tps=300)
        sr = comparator.evaluate_strategy(config, result)
        assert sr.meets_slo is False

    def test_compare_strategies(self):
        comparator = StrategyComparator(
            p95_latency_limit_ms=3000, min_throughput_tps=500
        )

        results = []
        configs_and_results = [
            ("baseline_static", 3500, 400),
            ("optimized_dynamic", 1500, 700),
            ("mid_config", 2500, 550),
        ]

        for name, lat, thr in configs_and_results:
            config = StrategyConfig(name=name)
            benchmark = _make_benchmark_result(
                name=name, p95_latency_ms=lat, throughput_tps=thr
            )
            sr = comparator.evaluate_strategy(config, benchmark)
            results.append(sr)

        report = comparator.compare_strategies(results)
        assert isinstance(report, ComparisonReport)
        assert report.best_strategy is not None
        assert report.best_strategy.config.name == "optimized_dynamic"

    def test_compare_empty_results(self):
        comparator = StrategyComparator()
        report = comparator.compare_strategies([])
        assert report.best_strategy is None

    def test_export_report(self):
        comparator = StrategyComparator(
            p95_latency_limit_ms=3000, min_throughput_tps=500
        )
        config = StrategyConfig(name="test")
        result = _make_benchmark_result()
        sr = comparator.evaluate_strategy(config, result)
        report = comparator.compare_strategies([sr])

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            filepath = f.name

        try:
            comparator.export_report(report, filepath)
            with open(filepath, "r") as f:
                data = json.load(f)
            assert "strategy_results" in data
            assert "best_strategy" in data
        finally:
            os.unlink(filepath)

    def test_comparison_report_to_dict(self):
        comparator = StrategyComparator()
        config = StrategyConfig(name="test")
        result = _make_benchmark_result()
        sr = comparator.evaluate_strategy(config, result)
        report = comparator.compare_strategies([sr])
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "strategy_results" in d
