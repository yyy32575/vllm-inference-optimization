"""
Workload Generator for VLLM Inference Optimization.

Constructs stress-test datasets based on real-world request length
distributions, enabling analysis of prefill/decode phase bottlenecks.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RequestSample:
    """A single benchmark request with input/output token specifications."""

    request_id: str
    prompt_tokens: int
    expected_output_tokens: int
    prompt_text: str = ""
    category: str = "medium"  # short / medium / long

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.expected_output_tokens


@dataclass
class WorkloadProfile:
    """Summary statistics for a generated workload."""

    total_requests: int
    avg_input_tokens: float
    avg_output_tokens: float
    p50_input_tokens: float
    p95_input_tokens: float
    p99_input_tokens: float
    p50_output_tokens: float
    p95_output_tokens: float
    p99_output_tokens: float
    short_request_ratio: float
    medium_request_ratio: float
    long_request_ratio: float
    estimated_prefill_heavy: float
    estimated_decode_heavy: float


class WorkloadGenerator:
    """
    Generates realistic workloads based on configurable length distributions.

    Supports log-normal, uniform, and bimodal distributions to model
    real-world request patterns. Classifies requests into short/medium/long
    categories for routing analysis.
    """

    def __init__(
        self,
        input_mean: float = 5.5,
        input_sigma: float = 1.0,
        input_min: int = 16,
        input_max: int = 2048,
        output_mean: float = 4.5,
        output_sigma: float = 0.8,
        output_min: int = 8,
        output_max: int = 1024,
        distribution: str = "lognormal",
        short_threshold: int = 256,
        long_threshold: int = 1024,
        seed: Optional[int] = None,
    ):
        self.input_mean = input_mean
        self.input_sigma = input_sigma
        self.input_min = input_min
        self.input_max = input_max
        self.output_mean = output_mean
        self.output_sigma = output_sigma
        self.output_min = output_min
        self.output_max = output_max
        self.distribution = distribution
        self.short_threshold = short_threshold
        self.long_threshold = long_threshold
        self.rng = np.random.RandomState(seed)

    def _sample_lengths(
        self, mean: float, sigma: float, min_val: int, max_val: int, count: int
    ) -> np.ndarray:
        """Sample token lengths from the configured distribution."""
        if self.distribution == "lognormal":
            samples = self.rng.lognormal(mean=mean, sigma=sigma, size=count)
        elif self.distribution == "uniform":
            samples = self.rng.uniform(low=min_val, high=max_val, size=count)
        elif self.distribution == "bimodal":
            # Mix of short and long requests
            n_short = count // 2
            n_long = count - n_short
            short_samples = self.rng.lognormal(
                mean=mean - 1.5, sigma=sigma * 0.5, size=n_short
            )
            long_samples = self.rng.lognormal(
                mean=mean + 0.5, sigma=sigma * 0.5, size=n_long
            )
            samples = np.concatenate([short_samples, long_samples])
            self.rng.shuffle(samples)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        samples = np.clip(samples, min_val, max_val).astype(int)
        return samples

    def _classify_request(self, input_tokens: int) -> str:
        """Classify request as short, medium, or long based on input length."""
        if input_tokens <= self.short_threshold:
            return "short"
        elif input_tokens >= self.long_threshold:
            return "long"
        return "medium"

    def _generate_prompt(self, num_tokens: int) -> str:
        """Generate a synthetic prompt with approximately the specified token count."""
        # Approximate 1 token ≈ 4 characters for English text
        words_needed = max(1, num_tokens * 3 // 4)
        base_words = [
            "The", "system", "processes", "data", "using", "advanced",
            "algorithms", "to", "optimize", "performance", "and", "reduce",
            "latency", "in", "distributed", "computing", "environments",
            "while", "maintaining", "high", "throughput", "across",
            "multiple", "nodes", "with", "efficient", "resource",
            "allocation", "strategies", "for", "maximum", "utilization",
        ]
        prompt_words = []
        for i in range(words_needed):
            prompt_words.append(base_words[i % len(base_words)])
        return " ".join(prompt_words)

    def generate(self, num_requests: int) -> List[RequestSample]:
        """Generate a workload of benchmark requests."""
        logger.info(
            "Generating %d requests with %s distribution",
            num_requests,
            self.distribution,
        )

        input_lengths = self._sample_lengths(
            self.input_mean,
            self.input_sigma,
            self.input_min,
            self.input_max,
            num_requests,
        )
        output_lengths = self._sample_lengths(
            self.output_mean,
            self.output_sigma,
            self.output_min,
            self.output_max,
            num_requests,
        )

        requests = []
        for i in range(num_requests):
            category = self._classify_request(int(input_lengths[i]))
            sample = RequestSample(
                request_id=f"req-{i:06d}",
                prompt_tokens=int(input_lengths[i]),
                expected_output_tokens=int(output_lengths[i]),
                prompt_text=self._generate_prompt(int(input_lengths[i])),
                category=category,
            )
            requests.append(sample)

        logger.info(
            "Generated %d requests: short=%d, medium=%d, long=%d",
            num_requests,
            sum(1 for r in requests if r.category == "short"),
            sum(1 for r in requests if r.category == "medium"),
            sum(1 for r in requests if r.category == "long"),
        )
        return requests

    def analyze_workload(self, requests: List[RequestSample]) -> WorkloadProfile:
        """Analyze workload characteristics and phase bottleneck estimates."""
        input_tokens = np.array([r.prompt_tokens for r in requests])
        output_tokens = np.array([r.expected_output_tokens for r in requests])

        categories = [r.category for r in requests]
        n = len(requests)

        # Estimate prefill vs decode dominance per request
        # Prefill-heavy: input >> output (prefill time dominates)
        # Decode-heavy: output >> input (decode iterations dominate)
        ratios = input_tokens / np.maximum(output_tokens, 1)
        prefill_heavy = float(np.mean(ratios > 2.0))
        decode_heavy = float(np.mean(ratios < 0.5))

        return WorkloadProfile(
            total_requests=n,
            avg_input_tokens=float(np.mean(input_tokens)),
            avg_output_tokens=float(np.mean(output_tokens)),
            p50_input_tokens=float(np.percentile(input_tokens, 50)),
            p95_input_tokens=float(np.percentile(input_tokens, 95)),
            p99_input_tokens=float(np.percentile(input_tokens, 99)),
            p50_output_tokens=float(np.percentile(output_tokens, 50)),
            p95_output_tokens=float(np.percentile(output_tokens, 95)),
            p99_output_tokens=float(np.percentile(output_tokens, 99)),
            short_request_ratio=categories.count("short") / n,
            medium_request_ratio=categories.count("medium") / n,
            long_request_ratio=categories.count("long") / n,
            estimated_prefill_heavy=prefill_heavy,
            estimated_decode_heavy=decode_heavy,
        )

    def split_by_category(
        self, requests: List[RequestSample]
    ) -> dict:
        """Split requests by category for routing analysis."""
        result = {"short": [], "medium": [], "long": []}
        for r in requests:
            result[r.category].append(r)
        return result

    def export_workload(
        self, requests: List[RequestSample], filepath: str
    ) -> None:
        """Export workload to a JSON file for reproducibility."""
        data = {
            "metadata": {
                "num_requests": len(requests),
                "distribution": self.distribution,
                "input_mean": self.input_mean,
                "input_sigma": self.input_sigma,
                "output_mean": self.output_mean,
                "output_sigma": self.output_sigma,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "requests": [
                {
                    "request_id": r.request_id,
                    "prompt_tokens": r.prompt_tokens,
                    "expected_output_tokens": r.expected_output_tokens,
                    "category": r.category,
                }
                for r in requests
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Exported workload to %s", filepath)

    @staticmethod
    def load_workload(filepath: str) -> List[RequestSample]:
        """Load a previously exported workload from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        requests = []
        for item in data["requests"]:
            requests.append(
                RequestSample(
                    request_id=item["request_id"],
                    prompt_tokens=item["prompt_tokens"],
                    expected_output_tokens=item["expected_output_tokens"],
                    category=item["category"],
                )
            )
        return requests
