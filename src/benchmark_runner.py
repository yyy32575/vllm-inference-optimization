"""
Benchmark Runner for VLLM Inference Optimization.

Executes benchmarks against a vLLM server, collecting latency,
throughput, and resource utilization metrics for each configuration.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

from .workload_generator import RequestSample

logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """Result metrics for a single completed request."""

    request_id: str
    prompt_tokens: int
    output_tokens: int
    time_to_first_token_ms: float
    total_latency_ms: float
    prefill_latency_ms: float
    decode_latency_ms: float
    tokens_per_second: float
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result for a single configuration run."""

    config_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_seconds: float
    # Latency metrics (ms)
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_ttft_ms: float
    p95_ttft_ms: float
    # Throughput metrics
    throughput_requests_per_sec: float
    throughput_tokens_per_sec: float
    avg_tokens_per_request: float
    # Phase breakdown
    avg_prefill_latency_ms: float
    avg_decode_latency_ms: float
    prefill_ratio: float  # fraction of latency in prefill
    # Resource utilization
    avg_gpu_utilization: Optional[float] = None
    peak_gpu_memory_gb: Optional[float] = None
    # Per-request details
    request_results: List[RequestResult] = field(default_factory=list)

    def meets_slo(
        self,
        p95_latency_limit_ms: float,
        min_throughput_tps: float,
    ) -> bool:
        """Check if this result meets the SLO constraints."""
        return (
            self.p95_latency_ms <= p95_latency_limit_ms
            and self.throughput_tokens_per_sec >= min_throughput_tps
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_name": self.config_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "avg_ttft_ms": round(self.avg_ttft_ms, 2),
            "p95_ttft_ms": round(self.p95_ttft_ms, 2),
            "throughput_requests_per_sec": round(
                self.throughput_requests_per_sec, 2
            ),
            "throughput_tokens_per_sec": round(
                self.throughput_tokens_per_sec, 2
            ),
            "avg_prefill_latency_ms": round(self.avg_prefill_latency_ms, 2),
            "avg_decode_latency_ms": round(self.avg_decode_latency_ms, 2),
            "prefill_ratio": round(self.prefill_ratio, 4),
        }


class BenchmarkRunner:
    """
    Runs inference benchmarks against a vLLM API server.

    Supports configurable concurrency levels, collects per-request
    metrics, and aggregates results including prefill/decode breakdown.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        api_endpoint: str = "/v1/completions",
        model: str = "meta-llama/Llama-2-7b-hf",
        timeout_seconds: int = 120,
    ):
        self.server_url = server_url.rstrip("/")
        self.api_endpoint = api_endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        request: RequestSample,
    ) -> RequestResult:
        """Send a single inference request and collect metrics."""
        url = f"{self.server_url}{self.api_endpoint}"
        payload = {
            "model": self.model,
            "prompt": request.prompt_text or "Hello",
            "max_tokens": request.expected_output_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        start_time = time.perf_counter()
        first_token_time = None
        output_tokens = 0

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with session.post(
                url, json=payload, timeout=timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return RequestResult(
                        request_id=request.request_id,
                        prompt_tokens=request.prompt_tokens,
                        output_tokens=0,
                        time_to_first_token_ms=0,
                        total_latency_ms=0,
                        prefill_latency_ms=0,
                        decode_latency_ms=0,
                        tokens_per_second=0,
                        success=False,
                        error=f"HTTP {response.status}: {error_text[:200]}",
                    )

                async for chunk in response.content.iter_any():
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    output_tokens += 1

            end_time = time.perf_counter()
            total_latency_ms = (end_time - start_time) * 1000
            ttft_ms = (
                (first_token_time - start_time) * 1000
                if first_token_time
                else total_latency_ms
            )
            decode_latency_ms = total_latency_ms - ttft_ms
            output_tokens = max(output_tokens, 1)
            tps = output_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0

            return RequestResult(
                request_id=request.request_id,
                prompt_tokens=request.prompt_tokens,
                output_tokens=output_tokens,
                time_to_first_token_ms=ttft_ms,
                total_latency_ms=total_latency_ms,
                prefill_latency_ms=ttft_ms,
                decode_latency_ms=decode_latency_ms,
                tokens_per_second=tps,
                success=True,
            )
        except Exception as e:
            end_time = time.perf_counter()
            return RequestResult(
                request_id=request.request_id,
                prompt_tokens=request.prompt_tokens,
                output_tokens=0,
                time_to_first_token_ms=0,
                total_latency_ms=(end_time - start_time) * 1000,
                prefill_latency_ms=0,
                decode_latency_ms=0,
                tokens_per_second=0,
                success=False,
                error=str(e),
            )

    async def _run_concurrent(
        self,
        requests: List[RequestSample],
        concurrency: int,
    ) -> List[RequestResult]:
        """Run requests with controlled concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def _limited_request(session, req):
            async with semaphore:
                return await self._send_request(session, req)

        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [_limited_request(session, req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    RequestResult(
                        request_id=requests[i].request_id,
                        prompt_tokens=requests[i].prompt_tokens,
                        output_tokens=0,
                        time_to_first_token_ms=0,
                        total_latency_ms=0,
                        prefill_latency_ms=0,
                        decode_latency_ms=0,
                        tokens_per_second=0,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)
        return final_results

    def _aggregate_results(
        self,
        config_name: str,
        request_results: List[RequestResult],
        total_duration: float,
    ) -> BenchmarkResult:
        """Aggregate per-request results into a benchmark summary."""
        successful = [r for r in request_results if r.success]
        if not successful:
            return BenchmarkResult(
                config_name=config_name,
                total_requests=len(request_results),
                successful_requests=0,
                failed_requests=len(request_results),
                total_duration_seconds=total_duration,
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
                request_results=request_results,
            )

        latencies = np.array([r.total_latency_ms for r in successful])
        ttfts = np.array([r.time_to_first_token_ms for r in successful])
        prefill_lats = np.array([r.prefill_latency_ms for r in successful])
        decode_lats = np.array([r.decode_latency_ms for r in successful])
        total_output_tokens = sum(r.output_tokens for r in successful)

        avg_prefill = float(np.mean(prefill_lats))
        avg_decode = float(np.mean(decode_lats))
        total_phase = avg_prefill + avg_decode

        return BenchmarkResult(
            config_name=config_name,
            total_requests=len(request_results),
            successful_requests=len(successful),
            failed_requests=len(request_results) - len(successful),
            total_duration_seconds=total_duration,
            avg_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            avg_ttft_ms=float(np.mean(ttfts)),
            p95_ttft_ms=float(np.percentile(ttfts, 95)),
            throughput_requests_per_sec=len(successful) / total_duration
            if total_duration > 0
            else 0,
            throughput_tokens_per_sec=total_output_tokens / total_duration
            if total_duration > 0
            else 0,
            avg_tokens_per_request=total_output_tokens / len(successful),
            avg_prefill_latency_ms=avg_prefill,
            avg_decode_latency_ms=avg_decode,
            prefill_ratio=avg_prefill / total_phase if total_phase > 0 else 0,
            request_results=request_results,
        )

    async def run_benchmark(
        self,
        requests: List[RequestSample],
        concurrency: int = 16,
        config_name: str = "default",
    ) -> BenchmarkResult:
        """Run a complete benchmark with the given requests and concurrency."""
        logger.info(
            "Starting benchmark '%s': %d requests, concurrency=%d",
            config_name,
            len(requests),
            concurrency,
        )

        start_time = time.perf_counter()
        results = await self._run_concurrent(requests, concurrency)
        end_time = time.perf_counter()

        total_duration = end_time - start_time
        benchmark_result = self._aggregate_results(
            config_name, results, total_duration
        )

        logger.info(
            "Benchmark '%s' complete: %d/%d successful, "
            "P95=%.1fms, throughput=%.1f tok/s",
            config_name,
            benchmark_result.successful_requests,
            benchmark_result.total_requests,
            benchmark_result.p95_latency_ms,
            benchmark_result.throughput_tokens_per_sec,
        )
        return benchmark_result

    def run_benchmark_sync(
        self,
        requests: List[RequestSample],
        concurrency: int = 16,
        config_name: str = "default",
    ) -> BenchmarkResult:
        """Synchronous wrapper for run_benchmark."""
        return asyncio.run(
            self.run_benchmark(requests, concurrency, config_name)
        )

    @staticmethod
    def compare_results(
        baseline: BenchmarkResult, optimized: BenchmarkResult
    ) -> Dict[str, Any]:
        """Compare two benchmark results and compute improvement metrics."""
        def safe_pct(base, opt):
            if base == 0:
                return 0.0
            return ((base - opt) / base) * 100

        return {
            "baseline": baseline.config_name,
            "optimized": optimized.config_name,
            "latency_improvement": {
                "p95_reduction_pct": round(
                    safe_pct(baseline.p95_latency_ms, optimized.p95_latency_ms),
                    2,
                ),
                "p99_reduction_pct": round(
                    safe_pct(baseline.p99_latency_ms, optimized.p99_latency_ms),
                    2,
                ),
                "avg_reduction_pct": round(
                    safe_pct(
                        baseline.avg_latency_ms, optimized.avg_latency_ms
                    ),
                    2,
                ),
            },
            "throughput_improvement": {
                "tokens_per_sec_increase_pct": round(
                    safe_pct(
                        optimized.throughput_tokens_per_sec,
                        baseline.throughput_tokens_per_sec,
                    ),
                    2,
                ),
                "requests_per_sec_increase_pct": round(
                    safe_pct(
                        optimized.throughput_requests_per_sec,
                        baseline.throughput_requests_per_sec,
                    ),
                    2,
                ),
            },
            "prefill_improvement_pct": round(
                safe_pct(
                    baseline.avg_prefill_latency_ms,
                    optimized.avg_prefill_latency_ms,
                ),
                2,
            ),
            "decode_improvement_pct": round(
                safe_pct(
                    baseline.avg_decode_latency_ms,
                    optimized.avg_decode_latency_ms,
                ),
                2,
            ),
        }
