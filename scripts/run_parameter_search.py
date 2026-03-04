#!/usr/bin/env python3
"""
SLO-Constrained Parameter Search for VLLM Inference Optimization.

Performs a systematic search over vLLM configuration parameters to
find the optimal configuration under SLO constraints.

Usage:
    python scripts/run_parameter_search.py [--config configs/default_config.yaml]
"""

import argparse
import json
import logging
import os
import sys

import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_runner import BenchmarkResult, BenchmarkRunner
from src.parameter_search import (
    ParameterSearch,
    SearchSpace,
    SLOConstraints,
)
from src.strategy_comparator import StrategyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_slo_profile(profile_path: str, profile_name: str) -> dict:
    """Load a specific SLO profile."""
    with open(profile_path, "r") as f:
        profiles = yaml.safe_load(f)
    return profiles.get("profiles", {}).get(profile_name, {})


def main():
    parser = argparse.ArgumentParser(
        description="VLLM SLO-Constrained Parameter Search"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--slo-profile",
        type=str,
        default="balanced",
        choices=["realtime", "batch", "balanced"],
        help="SLO profile to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate and display search configurations",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    slo_profile = load_slo_profile(
        "configs/slo_profiles.yaml", args.slo_profile
    )

    logger.info("=" * 60)
    logger.info("VLLM PARAMETER SEARCH")
    logger.info("SLO Profile: %s", args.slo_profile)
    logger.info("=" * 60)

    # Configure SLO constraints from profile
    constraints = slo_profile.get("constraints", {})
    slo = SLOConstraints(
        p95_latency_ms=constraints.get("p95_latency_ms", 3000),
        p99_latency_ms=constraints.get("p99_latency_ms", 10000),
        min_throughput_tokens_per_sec=constraints.get(
            "min_throughput_tokens_per_sec", 500
        ),
        max_ttft_ms=constraints.get("max_time_to_first_token_ms", 1000),
    )

    # Configure search space from config
    strategy_config = config.get("strategies", {})
    batching = strategy_config.get("batching", {})
    search_space = SearchSpace(
        max_num_batched_tokens=batching.get(
            "max_num_batched_tokens_options", [2048, 4096, 8192, 16384]
        ),
        max_num_seqs=batching.get("max_num_seqs_options", [64, 128, 256]),
        prefix_cache=strategy_config.get("prefix_cache", {}).get(
            "enabled_options", [True, False]
        ),
        batching_mode=batching.get("modes", ["dynamic", "static"]),
    )

    weight_latency = slo_profile.get("weight_latency", 0.5)
    weight_throughput = slo_profile.get("weight_throughput", 0.5)

    search = ParameterSearch(
        slo_constraints=slo,
        search_space=search_space,
        weight_latency=weight_latency,
        weight_throughput=weight_throughput,
    )

    # Generate configurations
    configs = search.generate_configs()
    logger.info(
        "Generated %d configurations (%d total combinations)",
        len(configs),
        search_space.total_combinations(),
    )

    if args.dry_run:
        logger.info("Dry run mode: displaying first 5 configurations")
        for c in configs[:5]:
            logger.info("  %s", json.dumps(c.to_dict(), indent=2))
        logger.info("... and %d more", max(0, len(configs) - 5))
        return

    # Live search requires running vLLM server
    logger.info(
        "Live parameter search requires a running vLLM server. "
        "Use --dry-run to preview configurations."
    )


if __name__ == "__main__":
    main()
