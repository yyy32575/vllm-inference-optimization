"""Tests for fault recovery module."""

import time

import pytest

from src.fault_recovery import (
    ConfigSnapshot,
    DegradationConfig,
    FallbackConfig,
    FaultRecoveryManager,
    HealthCheck,
    HealthThresholds,
    RateLimiter,
    RateLimiterConfig,
    RecoveryAction,
    ServiceHealth,
)


class TestServiceHealth:
    """Tests for ServiceHealth enum."""

    def test_health_values(self):
        assert ServiceHealth.HEALTHY.value == 0
        assert ServiceHealth.DEGRADED.value == 1
        assert ServiceHealth.CRITICAL.value == 2
        assert ServiceHealth.DOWN.value == 3


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allow_request_initial(self):
        limiter = RateLimiter(
            RateLimiterConfig(max_requests_per_sec=100, burst_size=10)
        )
        # Should allow initial burst
        for _ in range(10):
            assert limiter.allow_request() is True

    def test_reject_after_burst(self):
        limiter = RateLimiter(
            RateLimiterConfig(max_requests_per_sec=1, burst_size=2)
        )
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True
        # Third request should be rejected (burst exhausted)
        assert limiter.allow_request() is False

    def test_adjust_rate_healthy(self):
        config = RateLimiterConfig(
            max_requests_per_sec=100, recovery_factor=1.1, max_rate=500
        )
        limiter = RateLimiter(config)
        initial_rate = limiter.current_rate
        new_rate = limiter.adjust_rate(ServiceHealth.HEALTHY)
        assert new_rate >= initial_rate

    def test_adjust_rate_degraded(self):
        config = RateLimiterConfig(
            max_requests_per_sec=100, backoff_factor=0.5
        )
        limiter = RateLimiter(config)
        new_rate = limiter.adjust_rate(ServiceHealth.DEGRADED)
        assert new_rate == 50.0

    def test_adjust_rate_critical(self):
        config = RateLimiterConfig(
            max_requests_per_sec=100, min_rate=10
        )
        limiter = RateLimiter(config)
        new_rate = limiter.adjust_rate(ServiceHealth.CRITICAL)
        assert new_rate == 10.0

    def test_rate_bounded(self):
        config = RateLimiterConfig(
            max_requests_per_sec=100,
            min_rate=10,
            max_rate=200,
            recovery_factor=10.0,
        )
        limiter = RateLimiter(config)
        # Multiple recoveries should not exceed max
        for _ in range(20):
            limiter.adjust_rate(ServiceHealth.HEALTHY)
        assert limiter.current_rate <= config.max_rate

    def test_get_stats(self):
        limiter = RateLimiter(
            RateLimiterConfig(max_requests_per_sec=100, burst_size=5)
        )
        limiter.allow_request()
        limiter.allow_request()
        stats = limiter.get_stats()
        assert stats["total_allowed"] == 2
        assert "current_rate" in stats
        assert "rejection_rate" in stats


class TestFaultRecoveryManager:
    """Tests for FaultRecoveryManager."""

    def test_evaluate_healthy(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=1000,
            error_rate=0.01,
            throughput_fraction=0.9,
        )
        assert health.status == ServiceHealth.HEALTHY
        assert len(health.issues) == 0
        assert len(health.recommended_actions) == 0

    def test_evaluate_degraded_latency(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=7000,
            error_rate=0.01,
            throughput_fraction=0.9,
        )
        assert health.status == ServiceHealth.DEGRADED
        assert len(health.issues) > 0
        assert RecoveryAction.RATE_LIMIT in health.recommended_actions

    def test_evaluate_degraded_error_rate(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=1000,
            error_rate=0.10,
            throughput_fraction=0.9,
        )
        assert health.status == ServiceHealth.DEGRADED

    def test_evaluate_degraded_throughput(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=1000,
            error_rate=0.01,
            throughput_fraction=0.5,
        )
        assert health.status == ServiceHealth.DEGRADED

    def test_evaluate_critical_latency(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=15000,
            error_rate=0.01,
            throughput_fraction=0.9,
        )
        assert health.status == ServiceHealth.CRITICAL
        assert RecoveryAction.FALLBACK_MODEL in health.recommended_actions
        assert RecoveryAction.CONFIG_ROLLBACK in health.recommended_actions

    def test_evaluate_critical_error_rate(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=1000,
            error_rate=0.25,
            throughput_fraction=0.9,
        )
        assert health.status == ServiceHealth.CRITICAL

    def test_evaluate_critical_gpu_memory(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=1000,
            error_rate=0.01,
            throughput_fraction=0.9,
            gpu_memory_utilization=0.98,
        )
        assert health.status == ServiceHealth.CRITICAL

    def test_health_check_to_dict(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=7000,
            error_rate=0.10,
            throughput_fraction=0.5,
        )
        d = health.to_dict()
        assert "status" in d
        assert "issues" in d
        assert "recommended_actions" in d

    def test_get_degraded_config(self):
        manager = FaultRecoveryManager()
        config = manager.get_degraded_config()
        assert "max_tokens" in config
        assert "max_num_seqs" in config
        assert config["max_tokens"] < 1024

    def test_should_use_fallback(self):
        manager = FaultRecoveryManager()

        # Healthy - no fallback
        manager.evaluate_health(
            p95_latency_ms=1000, error_rate=0.01, throughput_fraction=0.9
        )
        assert manager.should_use_fallback() is False

        # Critical - use fallback
        manager.evaluate_health(
            p95_latency_ms=15000, error_rate=0.25, throughput_fraction=0.2
        )
        assert manager.should_use_fallback() is True

    def test_config_snapshot_and_rollback(self):
        manager = FaultRecoveryManager()

        # Save a healthy config
        good_config = {"max_num_seqs": 256, "max_num_batched_tokens": 8192}
        manager.current_health = ServiceHealth.HEALTHY
        manager.save_config_snapshot(good_config, "good config")

        # Simulate going to critical
        manager.current_health = ServiceHealth.CRITICAL
        manager.save_config_snapshot(
            {"max_num_seqs": 512}, "bad config"
        )

        # Rollback should return the healthy config
        rollback = manager.get_rollback_config()
        assert rollback is not None
        assert rollback["max_num_seqs"] == 256

    def test_rollback_no_healthy_config(self):
        manager = FaultRecoveryManager()
        manager.current_health = ServiceHealth.DEGRADED
        manager.save_config_snapshot({"x": 1}, "degraded config")
        rollback = manager.get_rollback_config()
        assert rollback is None

    def test_config_history_limit(self):
        manager = FaultRecoveryManager(max_config_history=3)
        for i in range(5):
            manager.save_config_snapshot({"i": i}, f"config {i}")
        assert len(manager.config_history) == 3

    def test_execute_recovery_degraded(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=7000,
            error_rate=0.10,
            throughput_fraction=0.5,
        )
        actions = manager.execute_recovery(health)
        assert len(actions) > 0
        action_types = [a["action"] for a in actions]
        assert "rate_limit" in action_types
        assert "degrade" in action_types

    def test_execute_recovery_critical(self):
        manager = FaultRecoveryManager()

        # Save a good config first
        manager.current_health = ServiceHealth.HEALTHY
        manager.save_config_snapshot({"good": True}, "baseline")

        health = manager.evaluate_health(
            p95_latency_ms=15000,
            error_rate=0.25,
            throughput_fraction=0.2,
        )
        actions = manager.execute_recovery(health)
        action_types = [a["action"] for a in actions]
        assert "rate_limit" in action_types
        assert "degrade" in action_types
        assert "fallback_model" in action_types
        assert "config_rollback" in action_types

    def test_action_log(self):
        manager = FaultRecoveryManager()
        health = manager.evaluate_health(
            p95_latency_ms=7000,
            error_rate=0.10,
            throughput_fraction=0.5,
        )
        manager.execute_recovery(health)
        log = manager.get_action_log()
        assert len(log) > 0

    def test_get_fallback_endpoint(self):
        config = FallbackConfig(
            fallback_server_url="http://localhost:8001"
        )
        manager = FaultRecoveryManager(fallback_config=config)
        assert manager.get_fallback_endpoint() == "http://localhost:8001"
