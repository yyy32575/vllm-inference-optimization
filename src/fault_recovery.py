"""
Fault Recovery Module for VLLM Inference Optimization.

Implements fault detection and recovery mechanisms including:
- Rate limiting under high load
- Graceful degradation strategies
- Small model fallback
- Configuration rollback
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Service health status levels (higher value = worse health)."""

    HEALTHY = 0
    DEGRADED = 1
    CRITICAL = 2
    DOWN = 3


class RecoveryAction(Enum):
    """Available recovery actions."""

    RATE_LIMIT = "rate_limit"
    DEGRADE = "degrade"
    FALLBACK_MODEL = "fallback_model"
    CONFIG_ROLLBACK = "config_rollback"
    RESTART = "restart"


@dataclass
class HealthThresholds:
    """Thresholds for health status determination."""

    # Latency thresholds (ms)
    degraded_p95_latency_ms: float = 5000.0
    critical_p95_latency_ms: float = 10000.0
    # Error rate thresholds
    degraded_error_rate: float = 0.05  # 5%
    critical_error_rate: float = 0.15  # 15%
    # Throughput thresholds (fraction of target)
    degraded_throughput_fraction: float = 0.7
    critical_throughput_fraction: float = 0.4
    # GPU thresholds
    critical_gpu_memory_utilization: float = 0.95
    critical_gpu_temperature_c: float = 85.0


@dataclass
class HealthCheck:
    """Result of a health check evaluation."""

    timestamp: float
    status: ServiceHealth
    p95_latency_ms: float
    error_rate: float
    throughput_fraction: float
    gpu_memory_utilization: float
    active_requests: int
    issues: List[str] = field(default_factory=list)
    recommended_actions: List[RecoveryAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "status": self.status.name.lower(),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "throughput_fraction": round(self.throughput_fraction, 4),
            "gpu_memory_utilization": round(self.gpu_memory_utilization, 4),
            "active_requests": self.active_requests,
            "issues": self.issues,
            "recommended_actions": [a.name.lower() for a in self.recommended_actions],
        }


@dataclass
class RateLimiterConfig:
    """Configuration for request rate limiting."""

    max_requests_per_sec: float = 100.0
    burst_size: int = 20
    backoff_factor: float = 0.5
    recovery_factor: float = 1.1
    min_rate: float = 10.0
    max_rate: float = 500.0


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""

    # Reduce max output tokens under load
    reduced_max_tokens: int = 256
    # Disable expensive features
    disable_prefix_cache: bool = True
    disable_beam_search: bool = True
    # Lower concurrency
    reduced_max_num_seqs: int = 32
    reduced_max_batched_tokens: int = 2048


@dataclass
class FallbackConfig:
    """Configuration for small model fallback."""

    primary_model: str = "meta-llama/Llama-2-7b-hf"
    fallback_model: str = "meta-llama/Llama-2-7b-hf"
    fallback_server_url: str = "http://localhost:8001"
    use_fallback_threshold: ServiceHealth = ServiceHealth.CRITICAL


@dataclass
class ConfigSnapshot:
    """Snapshot of a configuration for rollback."""

    timestamp: float
    config: Dict[str, Any]
    health_status: ServiceHealth
    description: str = ""


class RateLimiter:
    """
    Token bucket rate limiter with adaptive rate adjustment.

    Automatically reduces rate when errors increase and recovers
    when the system stabilizes.
    """

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        self.config = config or RateLimiterConfig()
        self.current_rate = self.config.max_requests_per_sec
        self.tokens = float(self.config.burst_size)
        self.last_refill_time = time.monotonic()
        self._total_allowed = 0
        self._total_rejected = 0

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        now = time.monotonic()
        elapsed = now - self.last_refill_time
        self.last_refill_time = now

        # Refill tokens
        self.tokens = min(
            float(self.config.burst_size),
            self.tokens + elapsed * self.current_rate,
        )

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self._total_allowed += 1
            return True
        self._total_rejected += 1
        return False

    def adjust_rate(self, health: ServiceHealth) -> float:
        """Adjust rate based on current health status."""
        if health == ServiceHealth.HEALTHY:
            self.current_rate = min(
                self.config.max_rate,
                self.current_rate * self.config.recovery_factor,
            )
        elif health == ServiceHealth.DEGRADED:
            self.current_rate *= self.config.backoff_factor
        elif health in (ServiceHealth.CRITICAL, ServiceHealth.DOWN):
            self.current_rate = self.config.min_rate

        self.current_rate = max(self.config.min_rate, self.current_rate)
        logger.info(
            "Rate adjusted to %.1f req/s (health=%s)",
            self.current_rate,
            health.name.lower(),
        )
        return self.current_rate

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        total = self._total_allowed + self._total_rejected
        return {
            "current_rate": round(self.current_rate, 2),
            "tokens_available": round(self.tokens, 2),
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
            "rejection_rate": (
                round(self._total_rejected / total, 4) if total > 0 else 0.0
            ),
        }


class FaultRecoveryManager:
    """
    Manages fault detection and recovery for vLLM inference services.

    Implements the fault recovery SOP:
    1. Detect anomalies via health checks
    2. Apply rate limiting to reduce load
    3. Degrade service quality to maintain availability
    4. Fall back to smaller model if needed
    5. Rollback configuration if issues persist
    """

    def __init__(
        self,
        thresholds: Optional[HealthThresholds] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
        degradation_config: Optional[DegradationConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
        max_config_history: int = 10,
    ):
        self.thresholds = thresholds or HealthThresholds()
        self.degradation_config = degradation_config or DegradationConfig()
        self.fallback_config = fallback_config or FallbackConfig()
        self.rate_limiter = RateLimiter(rate_limiter_config)
        self.config_history: List[ConfigSnapshot] = []
        self.max_config_history = max_config_history
        self.current_health = ServiceHealth.HEALTHY
        self._action_log: List[Dict[str, Any]] = []

    def evaluate_health(
        self,
        p95_latency_ms: float,
        error_rate: float,
        throughput_fraction: float,
        gpu_memory_utilization: float = 0.0,
        active_requests: int = 0,
    ) -> HealthCheck:
        """
        Evaluate current system health and recommend recovery actions.

        Returns a HealthCheck with the current status, detected issues,
        and recommended recovery actions.
        """
        issues = []
        actions = []
        status = ServiceHealth.HEALTHY

        # Check latency
        if p95_latency_ms > self.thresholds.critical_p95_latency_ms:
            issues.append(
                f"Critical P95 latency: {p95_latency_ms:.0f}ms "
                f"(limit: {self.thresholds.critical_p95_latency_ms:.0f}ms)"
            )
            status = ServiceHealth.CRITICAL
        elif p95_latency_ms > self.thresholds.degraded_p95_latency_ms:
            issues.append(
                f"High P95 latency: {p95_latency_ms:.0f}ms "
                f"(limit: {self.thresholds.degraded_p95_latency_ms:.0f}ms)"
            )
            if status.value < ServiceHealth.DEGRADED.value:
                status = ServiceHealth.DEGRADED

        # Check error rate
        if error_rate > self.thresholds.critical_error_rate:
            issues.append(
                f"Critical error rate: {error_rate:.1%} "
                f"(limit: {self.thresholds.critical_error_rate:.1%})"
            )
            status = ServiceHealth.CRITICAL
        elif error_rate > self.thresholds.degraded_error_rate:
            issues.append(
                f"High error rate: {error_rate:.1%} "
                f"(limit: {self.thresholds.degraded_error_rate:.1%})"
            )
            if status != ServiceHealth.CRITICAL:
                status = ServiceHealth.DEGRADED

        # Check throughput
        if throughput_fraction < self.thresholds.critical_throughput_fraction:
            issues.append(
                f"Critical throughput drop: {throughput_fraction:.1%} of target"
            )
            status = ServiceHealth.CRITICAL
        elif throughput_fraction < self.thresholds.degraded_throughput_fraction:
            issues.append(
                f"Low throughput: {throughput_fraction:.1%} of target"
            )
            if status != ServiceHealth.CRITICAL:
                status = ServiceHealth.DEGRADED

        # Check GPU memory
        if gpu_memory_utilization > self.thresholds.critical_gpu_memory_utilization:
            issues.append(
                f"GPU memory critical: {gpu_memory_utilization:.1%}"
            )
            status = ServiceHealth.CRITICAL

        # Determine recovery actions based on status
        if status == ServiceHealth.DEGRADED:
            actions.append(RecoveryAction.RATE_LIMIT)
            actions.append(RecoveryAction.DEGRADE)
        elif status == ServiceHealth.CRITICAL:
            actions.append(RecoveryAction.RATE_LIMIT)
            actions.append(RecoveryAction.DEGRADE)
            actions.append(RecoveryAction.FALLBACK_MODEL)
            actions.append(RecoveryAction.CONFIG_ROLLBACK)

        self.current_health = status
        self.rate_limiter.adjust_rate(status)

        return HealthCheck(
            timestamp=time.time(),
            status=status,
            p95_latency_ms=p95_latency_ms,
            error_rate=error_rate,
            throughput_fraction=throughput_fraction,
            gpu_memory_utilization=gpu_memory_utilization,
            active_requests=active_requests,
            issues=issues,
            recommended_actions=actions,
        )

    def get_degraded_config(self) -> Dict[str, Any]:
        """Get degraded configuration parameters for reducing load."""
        return {
            "max_tokens": self.degradation_config.reduced_max_tokens,
            "max_num_seqs": self.degradation_config.reduced_max_num_seqs,
            "max_num_batched_tokens": (
                self.degradation_config.reduced_max_batched_tokens
            ),
            "enable_prefix_caching": (
                not self.degradation_config.disable_prefix_cache
            ),
        }

    def get_fallback_endpoint(self) -> str:
        """Get the fallback model server URL."""
        return self.fallback_config.fallback_server_url

    def should_use_fallback(self) -> bool:
        """Check if the system should switch to fallback model."""
        critical_statuses = {ServiceHealth.CRITICAL, ServiceHealth.DOWN}
        return self.current_health in critical_statuses

    def save_config_snapshot(
        self, config: Dict[str, Any], description: str = ""
    ) -> None:
        """Save current configuration for potential rollback."""
        snapshot = ConfigSnapshot(
            timestamp=time.time(),
            config=config,
            health_status=self.current_health,
            description=description,
        )
        self.config_history.append(snapshot)
        if len(self.config_history) > self.max_config_history:
            self.config_history.pop(0)
        logger.info("Saved config snapshot: %s", description)

    def get_rollback_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the last known good configuration for rollback.

        Searches config history for the most recent healthy snapshot.
        """
        for snapshot in reversed(self.config_history):
            if snapshot.health_status == ServiceHealth.HEALTHY:
                logger.info(
                    "Found rollback target: %s (%.0f seconds ago)",
                    snapshot.description,
                    time.time() - snapshot.timestamp,
                )
                return snapshot.config
        return None

    def execute_recovery(
        self,
        health_check: HealthCheck,
    ) -> List[Dict[str, Any]]:
        """
        Execute recommended recovery actions from a health check.

        Returns a list of actions taken with their details.
        """
        actions_taken = []

        for action in health_check.recommended_actions:
            action_record = {
                "action": action.value,
                "timestamp": time.time(),
                "health_status": health_check.status.name.lower(),
            }

            if action == RecoveryAction.RATE_LIMIT:
                new_rate = self.rate_limiter.current_rate
                action_record["details"] = {
                    "new_rate": new_rate,
                    "stats": self.rate_limiter.get_stats(),
                }

            elif action == RecoveryAction.DEGRADE:
                degraded_config = self.get_degraded_config()
                action_record["details"] = {
                    "degraded_config": degraded_config,
                }

            elif action == RecoveryAction.FALLBACK_MODEL:
                action_record["details"] = {
                    "fallback_model": self.fallback_config.fallback_model,
                    "fallback_url": self.fallback_config.fallback_server_url,
                    "should_activate": self.should_use_fallback(),
                }

            elif action == RecoveryAction.CONFIG_ROLLBACK:
                rollback_config = self.get_rollback_config()
                action_record["details"] = {
                    "rollback_config": rollback_config,
                    "has_rollback_target": rollback_config is not None,
                }

            actions_taken.append(action_record)
            self._action_log.append(action_record)

        return actions_taken

    def get_action_log(self) -> List[Dict[str, Any]]:
        """Get the full action log."""
        return list(self._action_log)

    def export_recovery_report(self, filepath: str) -> None:
        """Export recovery action history to JSON."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        report = {
            "current_health": self.current_health.name.lower(),
            "rate_limiter_stats": self.rate_limiter.get_stats(),
            "config_snapshots": len(self.config_history),
            "action_log": self._action_log,
        }
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Exported recovery report to %s", filepath)
