#!/usr/bin/env python3
"""
Main benchmark entry point for VLLM Inference Optimization.

Runs a complete benchmark suite against a vLLM server, comparing
different optimization strategies and producing a detailed report.

Usage:
    python scripts/run_benchmark.py [--config configs/default_config.yaml]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_runner import BenchmarkRunner
from src.capacity_model import CapacityModel, GPUSpec, ModelSpec
from src.metrics_collector import MetricsCollector
from src.strategy_comparator import (
    ComparisonReport,
    StrategyComparator,
    StrategyConfig,
    create_baseline_config,
    create_optimized_config,
)
from src.workload_generator import WorkloadGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_capacity_analysis(config: dict) -> None:
    """Run capacity estimation analysis."""
    logger.info("=" * 60)
    logger.info("CAPACITY ANALYSIS")
    logger.info("=" * 60)

    gpu_config = config.get("capacity", {})
    gpu_spec = GPUSpec(
        name=gpu_config.get("gpu_type", "A100-80GB"),
        memory_gb=gpu_config.get("gpu_memory_gb", 80.0),
    )

    model = CapacityModel(
        gpu_spec=gpu_spec,
        safety_margin=gpu_config.get("safety_margin", 0.15),
    )

    # Memory breakdown
    mem = model.estimate_memory_breakdown()
    logger.info("Memory Breakdown:")
    for key, value in mem.items():
        logger.info("  %s: %.2f GB", key, value)

    # Capacity estimate
    estimate = model.estimate_capacity()
    logger.info("\nCapacity Estimate:")
    logger.info("  Max concurrent tokens: %d", estimate.max_concurrent_tokens)
    logger.info(
        "  Max concurrent sequences: %d", estimate.max_concurrent_sequences
    )
    logger.info("  Max requests/sec: %.2f", estimate.max_requests_per_sec)
    logger.info(
        "  Estimated throughput: %.2f tokens/sec",
        estimate.estimated_total_throughput_tps,
    )
    logger.info("  Bottleneck: %s", estimate.bottleneck)

    # Export results
    results_dir = config.get("monitoring", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    model.export_estimate(
        estimate, os.path.join(results_dir, "capacity_estimate.json")
    )

    return estimate


def generate_workload(config: dict):
    """Generate benchmark workload."""
    logger.info("=" * 60)
    logger.info("WORKLOAD GENERATION")
    logger.info("=" * 60)

    wl_config = config.get("workload", {})
    input_cfg = wl_config.get("input_length", {})
    output_cfg = wl_config.get("output_length", {})

    generator = WorkloadGenerator(
        input_mean=input_cfg.get("mean", 5.5),
        input_sigma=input_cfg.get("sigma", 1.0),
        input_min=input_cfg.get("min_tokens", 16),
        input_max=input_cfg.get("max_tokens", 2048),
        output_mean=output_cfg.get("mean", 4.5),
        output_sigma=output_cfg.get("sigma", 0.8),
        output_min=output_cfg.get("min_tokens", 8),
        output_max=output_cfg.get("max_tokens", 1024),
        distribution=input_cfg.get("distribution", "lognormal"),
        seed=42,
    )

    num_requests = wl_config.get("num_requests", 1000)
    requests = generator.generate(num_requests)
    profile = generator.analyze_workload(requests)

    logger.info("Workload Profile:")
    logger.info("  Total requests: %d", profile.total_requests)
    logger.info("  Avg input tokens: %.1f", profile.avg_input_tokens)
    logger.info("  Avg output tokens: %.1f", profile.avg_output_tokens)
    logger.info("  P95 input tokens: %.1f", profile.p95_input_tokens)
    logger.info(
        "  Short/Medium/Long: %.1f%% / %.1f%% / %.1f%%",
        profile.short_request_ratio * 100,
        profile.medium_request_ratio * 100,
        profile.long_request_ratio * 100,
    )
    logger.info(
        "  Prefill-heavy: %.1f%%, Decode-heavy: %.1f%%",
        profile.estimated_prefill_heavy * 100,
        profile.estimated_decode_heavy * 100,
    )

    # Export workload
    results_dir = config.get("monitoring", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    generator.export_workload(
        requests, os.path.join(results_dir, "workload.json")
    )

    return requests, profile


def main():
    parser = argparse.ArgumentParser(
        description="VLLM Inference Optimization Benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only run capacity analysis and workload generation (no live benchmark)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to use for benchmarking (default: 100)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("VLLM INFERENCE OPTIMIZATION BENCHMARK")
    logger.info("=" * 60)

    # Step 1: Capacity Analysis
    estimate = run_capacity_analysis(config)

    # Step 2: Generate Workload
    requests, profile = generate_workload(config)

    if args.analysis_only:
        logger.info("Analysis-only mode: skipping live benchmark")
        return

    # Step 3: Run Benchmarks (requires running vLLM server)
    server_config = config.get("server", {})
    server_url = (
        f"http://{server_config.get('host', 'localhost')}"
        f":{server_config.get('port', 8000)}"
    )

    runner = BenchmarkRunner(
        server_url=server_url,
        api_endpoint=server_config.get("api_endpoint", "/v1/completions"),
        model=server_config.get("model", "meta-llama/Llama-2-7b-hf"),
        timeout_seconds=server_config.get("timeout_seconds", 120),
    )

    slo_config = config.get("slo", {})
    comparator = StrategyComparator(
        p95_latency_limit_ms=slo_config.get("p95_latency_ms", 3000),
        min_throughput_tps=slo_config.get(
            "min_throughput_tokens_per_sec", 500
        ),
    )

    # Run baseline and optimized configurations
    configs = [create_baseline_config(), create_optimized_config()]
    strategy_results = []

    for strategy_config in configs:
        logger.info("Running benchmark with config: %s", strategy_config.name)
        result = runner.run_benchmark_sync(
            requests=requests[:args.num_requests],
            concurrency=16,
            config_name=strategy_config.name,
        )
        sr = comparator.evaluate_strategy(strategy_config, result)
        strategy_results.append(sr)

    # Compare results
    report = comparator.compare_strategies(strategy_results)

    results_dir = config.get("monitoring", {}).get("results_dir", "results")
    comparator.export_report(
        report, os.path.join(results_dir, "comparison_report.json")
    )

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    if report.best_strategy:
        logger.info("Best strategy: %s", report.best_strategy.config.name)
        logger.info("Best score: %.4f", report.best_strategy.score)
    if report.improvement_summary:
        logger.info("Improvements: %s", json.dumps(report.improvement_summary))


if __name__ == "__main__":
    main()
