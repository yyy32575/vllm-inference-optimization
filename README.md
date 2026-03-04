# VLLM 推理优化与容量建模

> 在固定GPU预算下同时优化吞吐与尾延迟，输出可落地的推理参数配置与容量评估方法

## 项目概述

本项目基于 vLLM 推理引擎，提供完整的推理性能优化与容量规划解决方案：

- **压测数据集生成**：基于真实长度分布构造请求，拆解 prefill/decode 两阶段瓶颈
- **策略对比评估**：动态/静态 batching、不同 max_num_batched_tokens、prefix cache 开关、长短请求分流
- **SLO约束参数搜索**：以 P95 时延与 tokens/s 为双目标的 Pareto 最优搜索
- **容量建模**：GPU 内存/算力/带宽瓶颈分析与最大并发估算
- **故障止血 SOP**：限流、降级、小模型兜底、配置回滚全流程
- **监控集成**：Prometheus 指标导出 + Grafana 仪表盘

## 实验结果

| 指标 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| 吞吐量 (tokens/s) | ~500 | ~650-700 | **+30%-40%** |
| P95 尾延迟 (ms) | ~4000 | ~2500-3000 | **-25%-35%** |
| GPU 利用率 | ~60% | ~75%-80% | **+15%-20%** |

## 项目结构

```
├── configs/                     # 配置文件
│   ├── default_config.yaml      # 默认压测配置
│   └── slo_profiles.yaml        # SLO约束配置
├── src/                         # 核心源代码
│   ├── workload_generator.py    # 压测数据集生成器
│   ├── benchmark_runner.py      # 基准测试运行器
│   ├── strategy_comparator.py   # 策略对比评估
│   ├── parameter_search.py      # SLO约束参数搜索
│   ├── capacity_model.py        # 容量建模
│   ├── metrics_collector.py     # Prometheus指标采集
│   └── fault_recovery.py        # 故障恢复模块
├── scripts/                     # 运行脚本
│   ├── run_benchmark.py         # 压测入口
│   └── run_parameter_search.py  # 参数搜索入口
├── monitoring/                  # 监控配置
│   ├── prometheus.yml           # Prometheus配置
│   └── grafana_dashboard.json   # Grafana仪表盘
├── sop/                         # 运维文档
│   └── fault_recovery_sop.md    # 故障止血SOP
├── tests/                       # 单元测试
│   ├── test_workload_generator.py
│   ├── test_benchmark_runner.py
│   ├── test_strategy_comparator.py
│   ├── test_parameter_search.py
│   ├── test_capacity_model.py
│   └── test_fault_recovery.py
├── requirements.txt
└── README.md
```

## 技术栈

- **推理引擎**：vLLM
- **GPU加速**：CUDA
- **监控**：Prometheus + Grafana
- **编程语言**：Python

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行容量分析与工作负载生成

```bash
python scripts/run_benchmark.py --analysis-only
```

### 运行参数搜索（预览模式）

```bash
python scripts/run_parameter_search.py --dry-run --slo-profile balanced
```

### 运行完整压测（需要 vLLM 服务）

```bash
# 1. 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000

# 2. 运行压测
python scripts/run_benchmark.py --config configs/default_config.yaml
```

### 运行测试

```bash
python -m pytest tests/ -v
```

## 核心模块说明

### 1. 压测数据集生成 (workload_generator.py)

基于对数正态分布构造真实请求长度分布，支持：
- 对数正态、均匀、双峰三种分布模式
- 自动分类短/中/长请求（可配置阈值）
- Prefill/Decode 阶段瓶颈分析
- 工作负载导出/加载（JSON格式）

### 2. 策略对比评估 (strategy_comparator.py)

网格搜索对比不同优化策略组合：
- **Batching模式**：动态 vs 静态
- **批处理上限**：2048 / 4096 / 8192 / 16384
- **并发序列数**：64 / 128 / 256
- **Prefix Cache**：开 / 关
- **请求路由**：长短请求分流

### 3. SLO约束参数搜索 (parameter_search.py)

双目标优化搜索：
- **目标1**：最小化 P95 延迟
- **目标2**：最大化吞吐量 (tokens/s)
- 支持三种 SLO 预设：realtime / batch / balanced
- Pareto前沿计算，识别非支配解
- SLO达标奖励机制

### 4. 容量建模 (capacity_model.py)

GPU资源评估与容量规划：
- 内存分解：模型权重 / KV Cache / 激活值 / 开销
- 最大并发 token 数和序列数估算
- Prefill (计算密集) / Decode (带宽密集) 吞吐估算
- 瓶颈识别：compute / memory / bandwidth
- 多GPU线性扩展估算

### 5. 故障恢复 (fault_recovery.py)

四级恢复机制：
1. **限流**：令牌桶自适应限流，Degraded 降 50%，Critical 降到最低
2. **降级**：减小输出长度、降低并发、缩小批处理
3. **小模型兜底**：流量切换到备用轻量模型
4. **配置回滚**：恢复到最近已知健康的配置快照

### 6. 监控指标 (metrics_collector.py)

Prometheus指标导出：
- 请求延迟直方图（total / prefill / decode）
- TTFT（首 token 时间）分布
- 吞吐量（tokens/s, requests/s）
- GPU利用率和显存使用
- SLO合规率和违规计数
- 批大小分布

## 配置说明

### SLO配置（configs/slo_profiles.yaml）

| 配置 | P95延迟 | 最小吞吐 | 适用场景 |
|------|---------|----------|---------|
| realtime | 1500ms | 200 tok/s | 在线聊天 |
| balanced | 5000ms | 500 tok/s | 通用服务 |
| batch | 15000ms | 1000 tok/s | 批量处理 |

## 故障止血SOP

详细操作流程参见 [sop/fault_recovery_sop.md](sop/fault_recovery_sop.md)

## License

MIT