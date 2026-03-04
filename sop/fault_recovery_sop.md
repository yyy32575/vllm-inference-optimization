# 故障止血SOP：VLLM 推理服务故障恢复标准操作流程

## 概述

本文档定义了 VLLM 推理服务在遇到性能下降、资源耗尽或服务故障时的标准操作流程（SOP）。
目标是在最短时间内恢复服务可用性，同时最小化对用户体验的影响。

## 健康状态定义

| 状态 | 含义 | P95延迟 | 错误率 | 吞吐量 |
|------|------|---------|--------|--------|
| ✅ Healthy | 服务正常 | < 5000ms | < 5% | > 70% target |
| ⚠️ Degraded | 性能下降 | 5000-10000ms | 5-15% | 40-70% target |
| 🔴 Critical | 严重故障 | > 10000ms | > 15% | < 40% target |
| ⛔ Down | 服务不可用 | N/A | 100% | 0 |

## 故障检测

### 监控指标
- **P95/P99 延迟**：通过 Prometheus `vllm_opt_request_latency_ms` 监控
- **错误率**：通过 `vllm_opt_requests_total{status="error"}` 监控
- **吞吐量**：通过 `vllm_opt_throughput_tokens_per_sec` 监控
- **GPU 利用率**：通过 `vllm_opt_gpu_utilization` 和 `vllm_opt_gpu_memory_used_gb` 监控
- **SLO 合规率**：通过 `vllm_opt_slo_compliance_ratio` 监控

### 告警规则
```yaml
# Prometheus alerting rules
groups:
  - name: vllm_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(vllm_opt_request_latency_ms_bucket{phase="total"}[5m])) > 5000
        for: 2m
        labels:
          severity: warning
      - alert: CriticalLatency
        expr: histogram_quantile(0.95, rate(vllm_opt_request_latency_ms_bucket{phase="total"}[5m])) > 10000
        for: 1m
        labels:
          severity: critical
      - alert: HighErrorRate
        expr: rate(vllm_opt_requests_total{status="error"}[5m]) / rate(vllm_opt_requests_total[5m]) > 0.15
        for: 1m
        labels:
          severity: critical
      - alert: GPUMemoryCritical
        expr: vllm_opt_gpu_memory_used_gb / 80 > 0.95
        for: 1m
        labels:
          severity: critical
```

## 恢复操作流程

### 第一步：限流（Rate Limiting）

**触发条件**：服务状态为 Degraded 或 Critical

**操作步骤**：
1. 启用自适应限流器，降低请求接入速率
2. 限流策略采用令牌桶算法：
   - Degraded：当前速率 × 0.5
   - Critical：降至最低速率（10 req/s）
3. 监控限流效果，观察延迟和错误率变化

**代码实现**：
```python
from src.fault_recovery import RateLimiter, RateLimiterConfig

limiter = RateLimiter(RateLimiterConfig(
    max_requests_per_sec=100.0,
    burst_size=20,
    backoff_factor=0.5,
    min_rate=10.0,
))

# 每个请求检查是否允许通过
if limiter.allow_request():
    # 处理请求
    pass
else:
    # 返回 429 Too Many Requests
    pass
```

**预期效果**：
- 减少系统负载 50-80%
- 延迟在 1-2 分钟内开始下降
- 错误率快速降低

### 第二步：降级（Degradation）

**触发条件**：限流后服务状态仍为 Degraded

**操作步骤**：
1. 降低最大输出长度：max_tokens 从 1024 降至 256
2. 降低并发度：max_num_seqs 从 256 降至 32
3. 降低批处理上限：max_num_batched_tokens 从 8192 降至 2048
4. 关闭 prefix cache（如果内存压力大）

**降级配置**：
```python
degraded_config = {
    "max_tokens": 256,           # 减少输出长度
    "max_num_seqs": 32,          # 降低并发
    "max_num_batched_tokens": 2048,  # 减小批处理
    "enable_prefix_caching": False,   # 关闭前缀缓存
}
```

**预期效果**：
- GPU 内存使用降低 30-50%
- P95 延迟下降 40-60%
- 吞吐量降低但服务稳定

### 第三步：小模型兜底（Fallback Model）

**触发条件**：服务状态为 Critical 且降级后仍未恢复

**操作步骤**：
1. 将流量切换到备用小模型服务
2. 小模型配置：
   - 主模型：Llama-2-7B → 备用模型：Llama-2-7B（或更小模型）
   - 备用服务运行在独立端口（默认 8001）
3. 通过负载均衡器或网关切换流量
4. 记录切换时间和影响范围

**切换逻辑**：
```python
from src.fault_recovery import FaultRecoveryManager

manager = FaultRecoveryManager()
if manager.should_use_fallback():
    fallback_url = manager.get_fallback_endpoint()
    # 将请求路由到 fallback_url
```

**预期效果**：
- 服务快速恢复（30秒内）
- 输出质量可能下降但可用性保证
- 为主模型修复争取时间

### 第四步：配置回滚（Config Rollback）

**触发条件**：配置变更后出现异常，需要回滚到上次已知良好的配置

**操作步骤**：
1. 从配置历史中获取最近的健康状态配置快照
2. 验证回滚配置的完整性
3. 应用回滚配置并重启服务
4. 监控恢复效果

**回滚流程**：
```python
from src.fault_recovery import FaultRecoveryManager

manager = FaultRecoveryManager()

# 保存当前配置（变更前）
manager.save_config_snapshot(current_config, "pre-change")

# 如需回滚
rollback_config = manager.get_rollback_config()
if rollback_config:
    # 应用回滚配置
    apply_config(rollback_config)
```

**预期效果**：
- 恢复到上次稳定状态
- 通常在 2-5 分钟内完成恢复

## 恢复优先级矩阵

| 故障类型 | 第一步 | 第二步 | 第三步 | 第四步 |
|----------|--------|--------|--------|--------|
| 延迟突增 | 限流 | 降级 | - | 配置回滚 |
| 错误率飙升 | 限流 | 配置回滚 | 小模型兜底 | - |
| GPU OOM | 降级 | 限流 | 配置回滚 | 小模型兜底 |
| 吞吐量暴跌 | 限流 | 降级 | 配置回滚 | 小模型兜底 |
| 服务完全不可用 | 小模型兜底 | 重启 | 配置回滚 | - |

## 恢复后检查

1. **确认指标恢复**：
   - P95 延迟回到 SLO 阈值以下
   - 错误率 < 5%
   - 吞吐量恢复到目标的 70% 以上

2. **逐步恢复正常配置**：
   - 每 5 分钟提升限流速率 10%
   - 观察指标稳定后恢复降级配置
   - 确认稳定 15 分钟后切回主模型

3. **事后复盘**：
   - 导出故障期间的恢复操作日志
   - 分析根因（流量突增/配置变更/资源不足/模型问题）
   - 更新告警阈值和恢复策略

## 联系人与升级流程

| 级别 | 响应时间 | 联系人 |
|------|---------|--------|
| Warning | 15分钟 | 值班工程师 |
| Critical | 5分钟 | 值班工程师 + 团队负责人 |
| Down | 立即 | 全体团队 + 管理层 |
