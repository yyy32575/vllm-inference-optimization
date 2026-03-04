"""Tests for workload generator module."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.workload_generator import RequestSample, WorkloadGenerator, WorkloadProfile


class TestRequestSample:
    """Tests for RequestSample dataclass."""

    def test_total_tokens(self):
        sample = RequestSample(
            request_id="req-000001",
            prompt_tokens=100,
            expected_output_tokens=50,
        )
        assert sample.total_tokens() == 150

    def test_default_category(self):
        sample = RequestSample(
            request_id="req-000001",
            prompt_tokens=100,
            expected_output_tokens=50,
        )
        assert sample.category == "medium"


class TestWorkloadGenerator:
    """Tests for WorkloadGenerator."""

    def test_generate_lognormal(self):
        gen = WorkloadGenerator(seed=42, distribution="lognormal")
        requests = gen.generate(100)
        assert len(requests) == 100
        assert all(isinstance(r, RequestSample) for r in requests)

    def test_generate_uniform(self):
        gen = WorkloadGenerator(seed=42, distribution="uniform")
        requests = gen.generate(50)
        assert len(requests) == 50

    def test_generate_bimodal(self):
        gen = WorkloadGenerator(seed=42, distribution="bimodal")
        requests = gen.generate(50)
        assert len(requests) == 50

    def test_invalid_distribution(self):
        gen = WorkloadGenerator(seed=42, distribution="invalid")
        with pytest.raises(ValueError, match="Unknown distribution"):
            gen.generate(10)

    def test_token_bounds(self):
        gen = WorkloadGenerator(
            seed=42,
            input_min=32,
            input_max=512,
            output_min=16,
            output_max=256,
        )
        requests = gen.generate(200)
        for r in requests:
            assert 32 <= r.prompt_tokens <= 512
            assert 16 <= r.expected_output_tokens <= 256

    def test_request_ids_unique(self):
        gen = WorkloadGenerator(seed=42)
        requests = gen.generate(100)
        ids = [r.request_id for r in requests]
        assert len(set(ids)) == len(ids)

    def test_classification_short(self):
        gen = WorkloadGenerator(short_threshold=256, long_threshold=1024)
        assert gen._classify_request(100) == "short"
        assert gen._classify_request(256) == "short"

    def test_classification_medium(self):
        gen = WorkloadGenerator(short_threshold=256, long_threshold=1024)
        assert gen._classify_request(500) == "medium"

    def test_classification_long(self):
        gen = WorkloadGenerator(short_threshold=256, long_threshold=1024)
        assert gen._classify_request(1024) == "long"
        assert gen._classify_request(2048) == "long"

    def test_deterministic_with_seed(self):
        gen1 = WorkloadGenerator(seed=123)
        gen2 = WorkloadGenerator(seed=123)
        r1 = gen1.generate(50)
        r2 = gen2.generate(50)
        for a, b in zip(r1, r2):
            assert a.prompt_tokens == b.prompt_tokens
            assert a.expected_output_tokens == b.expected_output_tokens

    def test_generate_prompt(self):
        gen = WorkloadGenerator(seed=42)
        prompt = gen._generate_prompt(100)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_split_by_category(self):
        gen = WorkloadGenerator(seed=42)
        requests = gen.generate(200)
        split = gen.split_by_category(requests)
        assert "short" in split
        assert "medium" in split
        assert "long" in split
        total = len(split["short"]) + len(split["medium"]) + len(split["long"])
        assert total == 200

    def test_analyze_workload(self):
        gen = WorkloadGenerator(seed=42)
        requests = gen.generate(500)
        profile = gen.analyze_workload(requests)

        assert isinstance(profile, WorkloadProfile)
        assert profile.total_requests == 500
        assert profile.avg_input_tokens > 0
        assert profile.avg_output_tokens > 0
        assert 0 <= profile.short_request_ratio <= 1
        assert 0 <= profile.medium_request_ratio <= 1
        assert 0 <= profile.long_request_ratio <= 1
        assert abs(
            profile.short_request_ratio
            + profile.medium_request_ratio
            + profile.long_request_ratio
            - 1.0
        ) < 1e-6
        assert profile.p50_input_tokens <= profile.p95_input_tokens
        assert profile.p95_input_tokens <= profile.p99_input_tokens

    def test_export_and_load_workload(self):
        gen = WorkloadGenerator(seed=42)
        requests = gen.generate(20)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            filepath = f.name

        try:
            gen.export_workload(requests, filepath)
            loaded = WorkloadGenerator.load_workload(filepath)
            assert len(loaded) == len(requests)
            for orig, load in zip(requests, loaded):
                assert orig.request_id == load.request_id
                assert orig.prompt_tokens == load.prompt_tokens
                assert orig.expected_output_tokens == load.expected_output_tokens
                assert orig.category == load.category
        finally:
            os.unlink(filepath)

    def test_export_workload_metadata(self):
        gen = WorkloadGenerator(seed=42, distribution="lognormal")
        requests = gen.generate(10)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            filepath = f.name

        try:
            gen.export_workload(requests, filepath)
            with open(filepath, "r") as f:
                data = json.load(f)
            assert "metadata" in data
            assert data["metadata"]["num_requests"] == 10
            assert data["metadata"]["distribution"] == "lognormal"
            assert "requests" in data
        finally:
            os.unlink(filepath)
