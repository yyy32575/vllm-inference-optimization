"""Tests for benchmark runner module."""

import pytest

from src.benchmark_runner import BenchmarkResult, BenchmarkRunner, RequestResult


class TestRequestResult:
    """Tests for RequestResult dataclass."""

    def test_success_result(self):
        result = RequestResult(
            request_id="req-000001",
            prompt_tokens=100,
            output_tokens=50,
            time_to_first_token_ms=200.0,
            total_latency_ms=1000.0,
            prefill_latency_ms=200.0,
            decode_latency_ms=800.0,
            tokens_per_second=50.0,
            success=True,
        )
        assert result.success is True
        assert result.error is None

    def test_error_result(self):
        result = RequestResult(
            request_id="req-000001",
            prompt_tokens=100,
            output_tokens=0,
            time_to_first_token_ms=0,
            total_latency_ms=5000.0,
            prefill_latency_ms=0,
            decode_latency_ms=0,
            tokens_per_second=0,
            success=False,
            error="Timeout",
        )
        assert result.success is False
        assert result.error == "Timeout"


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def _make_result(self, **kwargs) -> BenchmarkResult:
        defaults = dict(
            config_name="test",
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_duration_seconds=10.0,
            avg_latency_ms=1000.0,
            p50_latency_ms=800.0,
            p95_latency_ms=2000.0,
            p99_latency_ms=4000.0,
            avg_ttft_ms=300.0,
            p95_ttft_ms=500.0,
            throughput_requests_per_sec=9.5,
            throughput_tokens_per_sec=600.0,
            avg_tokens_per_request=63.0,
            avg_prefill_latency_ms=300.0,
            avg_decode_latency_ms=700.0,
            prefill_ratio=0.3,
        )
        defaults.update(kwargs)
        return BenchmarkResult(**defaults)

    def test_meets_slo_both(self):
        result = self._make_result(
            p95_latency_ms=2000, throughput_tokens_per_sec=600
        )
        assert result.meets_slo(3000, 500) is True

    def test_fails_slo_latency(self):
        result = self._make_result(
            p95_latency_ms=5000, throughput_tokens_per_sec=600
        )
        assert result.meets_slo(3000, 500) is False

    def test_fails_slo_throughput(self):
        result = self._make_result(
            p95_latency_ms=2000, throughput_tokens_per_sec=300
        )
        assert result.meets_slo(3000, 500) is False

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "config_name" in d
        assert "p95_latency_ms" in d
        assert "throughput_tokens_per_sec" in d
        assert "prefill_ratio" in d

    def test_to_dict_rounding(self):
        result = self._make_result(avg_latency_ms=1234.5678)
        d = result.to_dict()
        assert d["avg_latency_ms"] == 1234.57


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_init_default(self):
        runner = BenchmarkRunner()
        assert runner.server_url == "http://localhost:8000"
        assert runner.model == "meta-llama/Llama-2-7b-hf"

    def test_init_custom(self):
        runner = BenchmarkRunner(
            server_url="http://gpu-server:9000",
            api_endpoint="/v1/generate",
            model="custom-model",
            timeout_seconds=60,
        )
        assert runner.server_url == "http://gpu-server:9000"
        assert runner.model == "custom-model"
        assert runner.timeout_seconds == 60

    def test_compare_results(self):
        baseline = BenchmarkResult(
            config_name="baseline",
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            total_duration_seconds=10.0,
            avg_latency_ms=2000.0,
            p50_latency_ms=1800.0,
            p95_latency_ms=4000.0,
            p99_latency_ms=6000.0,
            avg_ttft_ms=500.0,
            p95_ttft_ms=800.0,
            throughput_requests_per_sec=10.0,
            throughput_tokens_per_sec=500.0,
            avg_tokens_per_request=50.0,
            avg_prefill_latency_ms=500.0,
            avg_decode_latency_ms=1500.0,
            prefill_ratio=0.25,
        )

        optimized = BenchmarkResult(
            config_name="optimized",
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            total_duration_seconds=7.0,
            avg_latency_ms=1200.0,
            p50_latency_ms=1000.0,
            p95_latency_ms=2500.0,
            p99_latency_ms=4000.0,
            avg_ttft_ms=300.0,
            p95_ttft_ms=500.0,
            throughput_requests_per_sec=14.3,
            throughput_tokens_per_sec=700.0,
            avg_tokens_per_request=49.0,
            avg_prefill_latency_ms=300.0,
            avg_decode_latency_ms=900.0,
            prefill_ratio=0.25,
        )

        comparison = BenchmarkRunner.compare_results(baseline, optimized)
        assert comparison["baseline"] == "baseline"
        assert comparison["optimized"] == "optimized"
        # P95 went from 4000 to 2500 = 37.5% reduction
        assert comparison["latency_improvement"]["p95_reduction_pct"] == 37.5
        # Throughput went from 500 to 700
        assert comparison["throughput_improvement"]["tokens_per_sec_increase_pct"] > 0

    def test_compare_results_zero_baseline(self):
        baseline = BenchmarkResult(
            config_name="empty",
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_duration_seconds=0,
            avg_latency_ms=0,
            p50_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            avg_ttft_ms=0,
            p95_ttft_ms=0,
            throughput_requests_per_sec=0,
            throughput_tokens_per_sec=0,
            avg_tokens_per_request=0,
            avg_prefill_latency_ms=0,
            avg_decode_latency_ms=0,
            prefill_ratio=0,
        )
        comparison = BenchmarkRunner.compare_results(baseline, baseline)
        assert comparison["latency_improvement"]["p95_reduction_pct"] == 0.0

    def test_aggregate_all_failed(self):
        """Test aggregation when all requests fail."""
        runner = BenchmarkRunner()
        failed_results = [
            RequestResult(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=0,
                time_to_first_token_ms=0,
                total_latency_ms=0,
                prefill_latency_ms=0,
                decode_latency_ms=0,
                tokens_per_second=0,
                success=False,
                error="Connection refused",
            )
            for i in range(5)
        ]
        result = runner._aggregate_results("test", failed_results, 5.0)
        assert result.successful_requests == 0
        assert result.failed_requests == 5
        assert result.throughput_tokens_per_sec == 0

    def test_aggregate_mixed_results(self):
        """Test aggregation with mixed success/failure."""
        runner = BenchmarkRunner()
        results = [
            RequestResult(
                request_id="req-0",
                prompt_tokens=100,
                output_tokens=50,
                time_to_first_token_ms=200.0,
                total_latency_ms=1000.0,
                prefill_latency_ms=200.0,
                decode_latency_ms=800.0,
                tokens_per_second=50.0,
                success=True,
            ),
            RequestResult(
                request_id="req-1",
                prompt_tokens=100,
                output_tokens=30,
                time_to_first_token_ms=300.0,
                total_latency_ms=1500.0,
                prefill_latency_ms=300.0,
                decode_latency_ms=1200.0,
                tokens_per_second=20.0,
                success=True,
            ),
            RequestResult(
                request_id="req-2",
                prompt_tokens=100,
                output_tokens=0,
                time_to_first_token_ms=0,
                total_latency_ms=5000.0,
                prefill_latency_ms=0,
                decode_latency_ms=0,
                tokens_per_second=0,
                success=False,
                error="Timeout",
            ),
        ]
        result = runner._aggregate_results("test", results, 5.0)
        assert result.successful_requests == 2
        assert result.failed_requests == 1
        assert result.throughput_tokens_per_sec > 0
        assert result.avg_prefill_latency_ms == 250.0  # (200+300)/2
