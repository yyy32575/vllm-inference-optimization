"""
Prometheus Metrics Collector for VLLM Inference Optimization.

Exports benchmark and inference metrics to Prometheus for monitoring
with Grafana dashboards.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    start_http_server,
)

from .benchmark_runner import BenchmarkResult, RequestResult

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and exports inference metrics to Prometheus.

    Tracks request latency, throughput, GPU utilization, and SLO
    compliance metrics for real-time monitoring.
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        namespace: str = "vllm_opt",
    ):
        self.registry = registry or CollectorRegistry()
        self.namespace = namespace
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        ns = self.namespace

        # Request metrics
        self.request_total = Counter(
            f"{ns}_requests_total",
            "Total number of inference requests",
            ["status", "category"],
            registry=self.registry,
        )

        self.request_latency = Histogram(
            f"{ns}_request_latency_ms",
            "Request latency in milliseconds",
            ["phase"],
            buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000],
            registry=self.registry,
        )

        self.ttft_latency = Histogram(
            f"{ns}_ttft_ms",
            "Time to first token in milliseconds",
            buckets=[50, 100, 200, 500, 1000, 2000, 5000],
            registry=self.registry,
        )

        # Throughput metrics
        self.tokens_generated = Counter(
            f"{ns}_tokens_generated_total",
            "Total tokens generated",
            registry=self.registry,
        )

        self.throughput_gauge = Gauge(
            f"{ns}_throughput_tokens_per_sec",
            "Current throughput in tokens per second",
            registry=self.registry,
        )

        self.requests_per_sec = Gauge(
            f"{ns}_requests_per_sec",
            "Current requests per second",
            registry=self.registry,
        )

        # Resource metrics
        self.gpu_utilization = Gauge(
            f"{ns}_gpu_utilization",
            "GPU utilization (0-1)",
            ["gpu_id"],
            registry=self.registry,
        )

        self.gpu_memory_used_gb = Gauge(
            f"{ns}_gpu_memory_used_gb",
            "GPU memory used in GB",
            ["gpu_id"],
            registry=self.registry,
        )

        self.kv_cache_usage = Gauge(
            f"{ns}_kv_cache_usage",
            "KV cache utilization (0-1)",
            registry=self.registry,
        )

        # SLO metrics
        self.slo_violations = Counter(
            f"{ns}_slo_violations_total",
            "Total SLO violations",
            ["constraint"],
            registry=self.registry,
        )

        self.slo_compliance_ratio = Gauge(
            f"{ns}_slo_compliance_ratio",
            "SLO compliance ratio (0-1)",
            registry=self.registry,
        )

        # Batch metrics
        self.batch_size = Histogram(
            f"{ns}_batch_size",
            "Number of requests per batch",
            buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
            registry=self.registry,
        )

        self.active_requests = Gauge(
            f"{ns}_active_requests",
            "Number of currently active requests",
            registry=self.registry,
        )

    def record_request(
        self, result: RequestResult, category: str = "medium"
    ) -> None:
        """Record metrics for a single completed request."""
        status = "success" if result.success else "error"
        self.request_total.labels(status=status, category=category).inc()

        if result.success:
            self.request_latency.labels(phase="total").observe(
                result.total_latency_ms
            )
            self.request_latency.labels(phase="prefill").observe(
                result.prefill_latency_ms
            )
            self.request_latency.labels(phase="decode").observe(
                result.decode_latency_ms
            )
            self.ttft_latency.observe(result.time_to_first_token_ms)
            self.tokens_generated.inc(result.output_tokens)

    def record_benchmark(self, result: BenchmarkResult) -> None:
        """Record aggregated benchmark result metrics."""
        self.throughput_gauge.set(result.throughput_tokens_per_sec)
        self.requests_per_sec.set(result.throughput_requests_per_sec)

        for req_result in result.request_results:
            self.record_request(req_result)

    def record_gpu_metrics(
        self,
        gpu_id: int,
        utilization: float,
        memory_used_gb: float,
    ) -> None:
        """Record GPU resource utilization metrics."""
        gpu_label = str(gpu_id)
        self.gpu_utilization.labels(gpu_id=gpu_label).set(utilization)
        self.gpu_memory_used_gb.labels(gpu_id=gpu_label).set(memory_used_gb)

    def record_slo_check(
        self,
        p95_latency_ms: float,
        throughput_tps: float,
        slo_p95_limit: float,
        slo_throughput_min: float,
    ) -> None:
        """Record SLO compliance check."""
        total_checks = 2
        violations = 0

        if p95_latency_ms > slo_p95_limit:
            self.slo_violations.labels(constraint="p95_latency").inc()
            violations += 1

        if throughput_tps < slo_throughput_min:
            self.slo_violations.labels(constraint="throughput").inc()
            violations += 1

        compliance = (total_checks - violations) / total_checks
        self.slo_compliance_ratio.set(compliance)

    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus exposition format."""
        return generate_latest(self.registry)

    def start_server(self, port: int = 9090) -> None:
        """Start a Prometheus metrics HTTP server."""
        start_http_server(port, registry=self.registry)
        logger.info("Prometheus metrics server started on port %d", port)
