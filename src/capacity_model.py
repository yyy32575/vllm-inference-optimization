"""
Capacity Model for VLLM Inference Optimization.

Estimates GPU resource requirements and maximum serving capacity
for given model configurations and workload characteristics.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .benchmark_runner import BenchmarkResult
from .workload_generator import WorkloadProfile

logger = logging.getLogger(__name__)


@dataclass
class GPUSpec:
    """GPU hardware specification."""

    name: str = "A100-80GB"
    memory_gb: float = 80.0
    compute_tflops_fp16: float = 312.0
    memory_bandwidth_gb_s: float = 2039.0
    tdp_watts: float = 300.0


@dataclass
class ModelSpec:
    """LLM model specification for capacity estimation."""

    name: str = "Llama-2-7B"
    parameters_billion: float = 7.0
    bytes_per_param: int = 2  # FP16
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    vocab_size: int = 32000
    max_sequence_length: int = 4096

    def model_size_gb(self) -> float:
        """Estimated model weight size in GB."""
        return self.parameters_billion * self.bytes_per_param

    def kv_cache_per_token_bytes(self) -> float:
        """KV cache memory per token per layer."""
        head_dim = self.hidden_size // self.num_attention_heads
        # 2 for K and V, bytes_per_param for dtype
        return (
            2 * self.num_attention_heads * head_dim * self.bytes_per_param
        )

    def kv_cache_per_token_all_layers_mb(self) -> float:
        """Total KV cache memory per token across all layers, in MB."""
        return (
            self.kv_cache_per_token_bytes()
            * self.num_layers
            / (1024 * 1024)
        )


@dataclass
class CapacityEstimate:
    """Capacity estimation result for a specific configuration."""

    gpu_spec: GPUSpec
    model_spec: ModelSpec
    # Memory breakdown
    model_memory_gb: float
    available_kv_cache_gb: float
    max_concurrent_tokens: int
    max_concurrent_sequences: int
    # Throughput estimates
    estimated_prefill_tokens_per_sec: float
    estimated_decode_tokens_per_sec: float
    estimated_total_throughput_tps: float
    # Capacity limits
    max_requests_per_sec: float
    recommended_concurrency: int
    gpu_utilization_estimate: float
    # Bottleneck analysis
    bottleneck: str  # "compute", "memory", or "bandwidth"
    headroom_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gpu": self.gpu_spec.name,
            "model": self.model_spec.name,
            "memory_breakdown": {
                "model_memory_gb": round(self.model_memory_gb, 2),
                "available_kv_cache_gb": round(self.available_kv_cache_gb, 2),
            },
            "capacity": {
                "max_concurrent_tokens": self.max_concurrent_tokens,
                "max_concurrent_sequences": self.max_concurrent_sequences,
                "max_requests_per_sec": round(self.max_requests_per_sec, 2),
                "recommended_concurrency": self.recommended_concurrency,
            },
            "throughput": {
                "prefill_tokens_per_sec": round(
                    self.estimated_prefill_tokens_per_sec, 2
                ),
                "decode_tokens_per_sec": round(
                    self.estimated_decode_tokens_per_sec, 2
                ),
                "total_throughput_tps": round(
                    self.estimated_total_throughput_tps, 2
                ),
            },
            "analysis": {
                "bottleneck": self.bottleneck,
                "gpu_utilization_estimate": round(
                    self.gpu_utilization_estimate, 4
                ),
                "headroom_pct": round(self.headroom_pct, 2),
            },
        }


class CapacityModel:
    """
    GPU capacity modeling for LLM inference.

    Estimates memory requirements, maximum concurrency, throughput limits,
    and identifies system bottlenecks for capacity planning.
    """

    def __init__(
        self,
        gpu_spec: Optional[GPUSpec] = None,
        model_spec: Optional[ModelSpec] = None,
        safety_margin: float = 0.15,
        gpu_memory_utilization: float = 0.90,
    ):
        self.gpu = gpu_spec or GPUSpec()
        self.model = model_spec or ModelSpec()
        self.safety_margin = safety_margin
        self.gpu_memory_utilization = gpu_memory_utilization

    def estimate_memory_breakdown(self) -> Dict[str, float]:
        """
        Estimate GPU memory allocation breakdown.

        Returns memory in GB for model weights, KV cache budget,
        activations, and overhead.
        """
        total_usable = self.gpu.memory_gb * self.gpu_memory_utilization
        model_mem = self.model.model_size_gb()

        # Activation memory estimate (~10% of model for typical batch sizes)
        activation_mem = model_mem * 0.10

        # CUDA overhead and framework memory
        overhead = 0.5  # ~500MB base overhead

        kv_cache_budget = max(
            0.0,
            total_usable - model_mem - activation_mem - overhead,
        )

        return {
            "total_gpu_memory_gb": self.gpu.memory_gb,
            "usable_memory_gb": total_usable,
            "model_weights_gb": model_mem,
            "activation_memory_gb": activation_mem,
            "overhead_gb": overhead,
            "kv_cache_budget_gb": kv_cache_budget,
        }

    def estimate_max_concurrent_tokens(self) -> int:
        """Estimate maximum number of tokens in KV cache simultaneously."""
        mem = self.estimate_memory_breakdown()
        kv_budget_bytes = mem["kv_cache_budget_gb"] * 1024 * 1024 * 1024
        bytes_per_token = (
            self.model.kv_cache_per_token_bytes() * self.model.num_layers
        )
        if bytes_per_token == 0:
            return 0
        max_tokens = int(kv_budget_bytes / bytes_per_token)
        # Apply safety margin
        max_tokens = int(max_tokens * (1.0 - self.safety_margin))
        return max_tokens

    def estimate_max_sequences(
        self, avg_sequence_length: int = 512
    ) -> int:
        """Estimate max concurrent sequences given average length."""
        max_tokens = self.estimate_max_concurrent_tokens()
        if avg_sequence_length == 0:
            return 0
        return max(1, max_tokens // avg_sequence_length)

    def estimate_throughput(
        self, workload_profile: Optional[WorkloadProfile] = None
    ) -> Dict[str, float]:
        """
        Estimate throughput for prefill and decode phases.

        Prefill is compute-bound, decode is memory-bandwidth-bound.
        """
        # Prefill: compute-bound (matrix multiplication)
        # Approximate: FLOPs per token = 2 * model_params
        flops_per_token = 2 * self.model.parameters_billion * 1e9
        gpu_flops = self.gpu.compute_tflops_fp16 * 1e12
        # Typical utilization for prefill: ~50-70%
        prefill_utilization = 0.60
        prefill_tps = (gpu_flops * prefill_utilization) / flops_per_token

        # Decode: memory-bandwidth-bound (loading model weights per token)
        model_bytes = self.model.model_size_gb() * 1024 * 1024 * 1024
        bandwidth = self.gpu.memory_bandwidth_gb_s * 1024 * 1024 * 1024
        # Typical utilization for decode: ~60-80%
        decode_utilization = 0.70
        if model_bytes == 0:
            decode_tps = 0.0
        else:
            decode_tps = (bandwidth * decode_utilization) / model_bytes

        # Effective throughput depends on workload mix
        if workload_profile:
            prefill_ratio = workload_profile.estimated_prefill_heavy
            decode_ratio = workload_profile.estimated_decode_heavy
            balanced = 1.0 - prefill_ratio - decode_ratio
            effective_tps = (
                prefill_tps * prefill_ratio
                + decode_tps * decode_ratio
                + (prefill_tps + decode_tps) / 2 * balanced
            )
        else:
            effective_tps = (prefill_tps + decode_tps) / 2

        return {
            "prefill_tokens_per_sec": prefill_tps,
            "decode_tokens_per_sec": decode_tps,
            "effective_throughput_tps": effective_tps,
        }

    def identify_bottleneck(self) -> str:
        """Identify the primary system bottleneck."""
        throughput = self.estimate_throughput()
        max_tokens = self.estimate_max_concurrent_tokens()
        mem = self.estimate_memory_breakdown()

        # If KV cache is very limited, memory is the bottleneck
        if mem["kv_cache_budget_gb"] < 2.0:
            return "memory"

        # If decode throughput is significantly lower, it's bandwidth-bound
        if (
            throughput["decode_tokens_per_sec"]
            < throughput["prefill_tokens_per_sec"] * 0.5
        ):
            return "bandwidth"

        return "compute"

    def estimate_capacity(
        self,
        workload_profile: Optional[WorkloadProfile] = None,
        avg_sequence_length: int = 512,
        avg_output_length: int = 128,
    ) -> CapacityEstimate:
        """
        Produce a complete capacity estimate for the current configuration.

        Combines memory analysis, throughput estimation, and bottleneck
        identification into a comprehensive capacity assessment.
        """
        mem = self.estimate_memory_breakdown()
        max_tokens = self.estimate_max_concurrent_tokens()
        max_seqs = self.estimate_max_sequences(avg_sequence_length)
        throughput = self.estimate_throughput(workload_profile)
        bottleneck = self.identify_bottleneck()

        # Estimate max requests per second
        if avg_output_length > 0:
            max_rps = throughput["effective_throughput_tps"] / avg_output_length
        else:
            max_rps = 0.0

        # Recommended concurrency (aim for 70-80% utilization)
        target_utilization = 0.75
        recommended_concurrency = max(
            1, int(max_seqs * target_utilization)
        )

        # GPU utilization estimate
        gpu_util = min(
            1.0,
            mem["model_weights_gb"] / mem["usable_memory_gb"] * 0.7
            + 0.3 * target_utilization,
        )

        # Headroom
        headroom = max(
            0.0,
            (mem["kv_cache_budget_gb"] / max(mem["usable_memory_gb"], 1)) * 100,
        )

        return CapacityEstimate(
            gpu_spec=self.gpu,
            model_spec=self.model,
            model_memory_gb=mem["model_weights_gb"],
            available_kv_cache_gb=mem["kv_cache_budget_gb"],
            max_concurrent_tokens=max_tokens,
            max_concurrent_sequences=max_seqs,
            estimated_prefill_tokens_per_sec=throughput["prefill_tokens_per_sec"],
            estimated_decode_tokens_per_sec=throughput["decode_tokens_per_sec"],
            estimated_total_throughput_tps=throughput["effective_throughput_tps"],
            max_requests_per_sec=max_rps,
            recommended_concurrency=recommended_concurrency,
            gpu_utilization_estimate=gpu_util,
            bottleneck=bottleneck,
            headroom_pct=headroom,
        )

    def scale_estimate(
        self,
        num_gpus: int,
        capacity: CapacityEstimate,
    ) -> Dict[str, Any]:
        """Estimate capacity when scaling to multiple GPUs."""
        # Linear scaling with efficiency factor
        efficiency = 0.85 if num_gpus <= 4 else 0.75
        return {
            "num_gpus": num_gpus,
            "scaling_efficiency": efficiency,
            "total_throughput_tps": round(
                capacity.estimated_total_throughput_tps * num_gpus * efficiency,
                2,
            ),
            "total_max_rps": round(
                capacity.max_requests_per_sec * num_gpus * efficiency, 2
            ),
            "total_max_concurrent_sequences": int(
                capacity.max_concurrent_sequences * num_gpus
            ),
        }

    def export_estimate(
        self, estimate: CapacityEstimate, filepath: str
    ) -> None:
        """Export capacity estimate to JSON."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(estimate.to_dict(), f, indent=2)
        logger.info("Exported capacity estimate to %s", filepath)
