"""Tests for capacity model module."""

import json
import os
import tempfile

import pytest

from src.capacity_model import (
    CapacityEstimate,
    CapacityModel,
    GPUSpec,
    ModelSpec,
)
from src.workload_generator import WorkloadProfile


class TestGPUSpec:
    """Tests for GPUSpec."""

    def test_default_spec(self):
        gpu = GPUSpec()
        assert gpu.name == "A100-80GB"
        assert gpu.memory_gb == 80.0
        assert gpu.compute_tflops_fp16 > 0


class TestModelSpec:
    """Tests for ModelSpec."""

    def test_default_spec(self):
        model = ModelSpec()
        assert model.name == "Llama-2-7B"
        assert model.parameters_billion == 7.0

    def test_model_size(self):
        model = ModelSpec(parameters_billion=7.0, bytes_per_param=2)
        assert model.model_size_gb() == 14.0

    def test_kv_cache_per_token(self):
        model = ModelSpec(
            hidden_size=4096,
            num_attention_heads=32,
            bytes_per_param=2,
            num_layers=32,
        )
        per_token = model.kv_cache_per_token_bytes()
        assert per_token > 0
        per_token_all = model.kv_cache_per_token_all_layers_mb()
        assert per_token_all > 0


class TestCapacityModel:
    """Tests for CapacityModel."""

    def test_memory_breakdown(self):
        model = CapacityModel()
        mem = model.estimate_memory_breakdown()

        assert "total_gpu_memory_gb" in mem
        assert "usable_memory_gb" in mem
        assert "model_weights_gb" in mem
        assert "kv_cache_budget_gb" in mem
        assert mem["kv_cache_budget_gb"] > 0
        assert mem["usable_memory_gb"] <= mem["total_gpu_memory_gb"]
        total_allocated = (
            mem["model_weights_gb"]
            + mem["activation_memory_gb"]
            + mem["overhead_gb"]
            + mem["kv_cache_budget_gb"]
        )
        assert abs(total_allocated - mem["usable_memory_gb"]) < 0.01

    def test_max_concurrent_tokens(self):
        model = CapacityModel()
        max_tokens = model.estimate_max_concurrent_tokens()
        assert max_tokens > 0
        assert isinstance(max_tokens, int)

    def test_max_sequences(self):
        model = CapacityModel()
        max_seqs = model.estimate_max_sequences(avg_sequence_length=512)
        assert max_seqs > 0
        assert isinstance(max_seqs, int)

    def test_max_sequences_zero_length(self):
        model = CapacityModel()
        max_seqs = model.estimate_max_sequences(avg_sequence_length=0)
        assert max_seqs == 0

    def test_throughput_estimate(self):
        model = CapacityModel()
        throughput = model.estimate_throughput()
        assert throughput["prefill_tokens_per_sec"] > 0
        assert throughput["decode_tokens_per_sec"] > 0
        assert throughput["effective_throughput_tps"] > 0

    def test_throughput_with_workload_profile(self):
        model = CapacityModel()
        profile = WorkloadProfile(
            total_requests=1000,
            avg_input_tokens=300,
            avg_output_tokens=100,
            p50_input_tokens=250,
            p95_input_tokens=800,
            p99_input_tokens=1500,
            p50_output_tokens=80,
            p95_output_tokens=250,
            p99_output_tokens=500,
            short_request_ratio=0.5,
            medium_request_ratio=0.3,
            long_request_ratio=0.2,
            estimated_prefill_heavy=0.3,
            estimated_decode_heavy=0.4,
        )
        throughput = model.estimate_throughput(workload_profile=profile)
        assert throughput["effective_throughput_tps"] > 0

    def test_identify_bottleneck(self):
        model = CapacityModel()
        bottleneck = model.identify_bottleneck()
        assert bottleneck in ("compute", "memory", "bandwidth")

    def test_capacity_estimate(self):
        model = CapacityModel()
        estimate = model.estimate_capacity()

        assert isinstance(estimate, CapacityEstimate)
        assert estimate.max_concurrent_tokens > 0
        assert estimate.max_concurrent_sequences > 0
        assert estimate.estimated_total_throughput_tps > 0
        assert estimate.max_requests_per_sec > 0
        assert estimate.recommended_concurrency > 0
        assert estimate.bottleneck in ("compute", "memory", "bandwidth")
        assert 0 <= estimate.gpu_utilization_estimate <= 1.0

    def test_capacity_estimate_to_dict(self):
        model = CapacityModel()
        estimate = model.estimate_capacity()
        d = estimate.to_dict()
        assert "gpu" in d
        assert "model" in d
        assert "memory_breakdown" in d
        assert "capacity" in d
        assert "throughput" in d
        assert "analysis" in d

    def test_scale_estimate(self):
        model = CapacityModel()
        estimate = model.estimate_capacity()

        # Single GPU
        scale_1 = model.scale_estimate(1, estimate)
        assert scale_1["num_gpus"] == 1

        # Multi GPU should have higher throughput
        scale_4 = model.scale_estimate(4, estimate)
        assert scale_4["total_throughput_tps"] > scale_1["total_throughput_tps"]
        assert scale_4["total_max_rps"] > scale_1["total_max_rps"]

    def test_scale_efficiency_decreases(self):
        model = CapacityModel()
        estimate = model.estimate_capacity()

        scale_4 = model.scale_estimate(4, estimate)
        scale_8 = model.scale_estimate(8, estimate)
        # Efficiency should be lower for more GPUs
        assert scale_8["scaling_efficiency"] <= scale_4["scaling_efficiency"]

    def test_small_gpu(self):
        """Test with a smaller GPU to verify memory constraints."""
        small_gpu = GPUSpec(name="T4-16GB", memory_gb=16.0)
        model = CapacityModel(gpu_spec=small_gpu)
        mem = model.estimate_memory_breakdown()
        # Should still have valid (possibly small) KV cache budget
        assert mem["kv_cache_budget_gb"] >= 0

    def test_export_estimate(self):
        model = CapacityModel()
        estimate = model.estimate_capacity()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            filepath = f.name

        try:
            model.export_estimate(estimate, filepath)
            with open(filepath, "r") as f:
                data = json.load(f)
            assert "gpu" in data
            assert "capacity" in data
        finally:
            os.unlink(filepath)

    def test_different_model_specs(self):
        """Test capacity model with different model sizes."""
        # 13B model
        model_13b = ModelSpec(
            name="Llama-2-13B",
            parameters_billion=13.0,
            num_layers=40,
            hidden_size=5120,
            num_attention_heads=40,
        )
        cap_13b = CapacityModel(model_spec=model_13b)
        est_13b = cap_13b.estimate_capacity()

        # 7B model
        model_7b = ModelSpec()
        cap_7b = CapacityModel(model_spec=model_7b)
        est_7b = cap_7b.estimate_capacity()

        # 7B should have more room for KV cache
        assert est_7b.available_kv_cache_gb > est_13b.available_kv_cache_gb
