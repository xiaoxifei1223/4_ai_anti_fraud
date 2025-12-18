# 模块三：Analytics 数据监控与挖掘模块详细设计

> 版本：v0.1（设计草案）  
> 关联模块：模块一（快速反欺诈）、模块二（LLM+Agent 智能反欺诈）

---

## 1. 背景与目标

### 1.1 背景

现有系统已经实现：

- **模块一：快速反欺诈（Fast 模块）**  
  - 技术栈：XGBoost + 45 维特征（30 表格特征 + 15 图特征）+ 规则引擎  
  - 能力：毫秒级在线打分与决策（`fast_detect`），支持高 QPS 实时拦截

- **模块二：LLM+Agent 智能反欺诈（LLM 模块）**  
  - 技术栈：Kimi 2（OpenAI 兼容接口）+ LangGraph 多 Agent 工作流  
  - 能力：对灰区/高风险样本做多维度深度分析，提供结构化 JSON 决策和自然语言解释

当前仍缺少一个**系统级的数据分析与监控模块**，用于：

1. 从整体视角监控系统运行状态（实时监控、告警线索）
2. 分析模型与规则的长期表现（召回率、误报率、规则贡献度）
3. 审计 LLM+Agent 的决策行为（可解释、可复盘、可合规审查）

### 1.2 模块定位

**模块三（Analytics）** 是一个“数据趋势 + 监控 + 挖掘”的分析子系统，主要回答：

- **实时监控**：现在系统运行得怎么样？  
  - QPS、拒绝率、人工审核率是否异常？  
  - 各渠道/地区/产品的风险情况如何？
- **模型 & 规则表现**：模型/规则是否老化或“过猛”？  
  - fast 模块 / LLM 模块 / 融合结果在召回率和误报率上的表现如何？  
  - 哪些规则贡献大、误杀多？
- **LLM/Agent 决策审计**：第二模块是否可复盘、可解释？  
  - 单笔交易的完整决策链路是什么？  
  - 各 Agent（Behavior/Graph/Rule/Reflection/Judge）之间是否一致？

> 模块三不参与在线判决路径，只做 **监控 + 评估 + 挖掘**，允许秒级甚至分钟级延迟。

---

## 2. 阶段划分（Phase 1 & Phase 2）

### 2.1 Phase 1：实时监控与基础趋势

**目标**：

- 先搭建一套“能看得见”的监控视图，优先保障：
  - 最近 N 分钟内整体交易态势（总量/拒绝/审核）
  - fast/llm 两个模块的触发比例和基础性能指标
  - 按渠道（channel）、地区（region）等维度拆分的基础趋势

**交付内容（后端）**：

- 实时总览接口：`GET /api/v1/analytics/realtime/summary`
- 按维度拆分统计：`GET /api/v1/analytics/realtime/by-dimension`
- 时间序列趋势接口：`GET /api/v1/analytics/realtime/timeseries`

### 2.2 Phase 2：模型&规则表现 + LLM/Agent 审计

**目标**：

- 在 Phase 1 的基础上，深度评估模型、规则与 LLM 的实际表现，支持 AB 测试和策略/模型迭代。

**交付内容（后端）**：

- 模型表现接口：`GET /api/v1/analytics/model/performance`
- 规则表现接口：`GET /api/v1/analytics/rules/stats`
- LLM 决策链路审计接口：`GET /api/v1/analytics/llm/trace`
- Agent 统计与一致性接口（可选）：`GET /api/v1/analytics/llm/agents/summary`

> 后续可以在此基础上扩展 Phase 3：离线挖掘（模式发现、模型漂移检测等），此处暂不展开实现细节。

---

## 3. 总体架构与与模块 1/2 的集成关系

### 3.1 逻辑架构

- **数据生产层**：
  - 模块 1 的 `fast_detect` + `_save_transaction` 写入交易记录
  - 模块 2 的 `llm_detect` + `LlmAgentService` 写入 LLM 相关结果

- **存储层**：
  - `transactions` 表（主事实表）
  - 现有 `DetectionLog`/日志表（如存在）
  - 后续可引入统计表（如 `model_daily_stats`、`rule_stats`）用于长周期聚合

- **分析服务层（模块三）**：
  - 新增 `analytics_service`（服务层）
  - 新增 `analytics` API 路由（FastAPI 路由模块）

- **展示层**（前端）：
  - 实时监控大屏（Ant Design）
  - 模型 & 规则表现面板
  - LLM/Agent 审计面板

### 3.2 与模块 1/2 的数据交互关系

- **模块一 → 模块三**：
  - `_save_transaction` 写入：
    - 基本交易信息：`transaction_id`, `user_id`, `amount`, `transaction_type`, `merchant_id`, `device_id`, `ip_address`, `location`, `created_at` 等
    - fast 模块结果：`fraud_score`, `risk_level`, `detection_method='fast'`
    - 规则结果：`rule_decision`, `rule_reasons`, `triggered_rules`（JSON）
  - 模块三基于以上数据做：
    - 实时监控（Phase 1）
    - 模型表现分析（Phase 2 部分）
    - 规则表现分析（Phase 2 部分）

- **模块二 → 模块三**：
  - `llm_detect` 在 fast 结果基础上追加/更新交易记录：
    - `detection_method='llm'`
    - `llm_decision`, `llm_risk_score`, `llm_confidence`
    - `llm_analysis`, `llm_reasoning`
  - `llm_service_v2` 在 JudgeAgent 阶段额外构造：
    - `llm_agents_snapshot` 字段（JSON），包含：
      ```json
      {
        "coordinator": {...},
        "behavior": {...},
        "graph": {...},
        "rule": {...},
        "reflection": {...},
        "judge": {...}
      }
      ```
  - 模块三基于以上数据做：
    - LLM vs fast vs 融合的模型表现分析
    - 单笔交易的 LLM/Agent 决策审计视图

> 设计原则：模块三**只读**已有数据，不影响模块一/二 的在线性能路径。

---

## 4. 数据模型与字段规划

### 4.1 `transactions` 表（关键字段视图）

> 实际字段以 ORM/数据库定义为准，这里给出逻辑视图。

- **主键 & 基本信息**：
  - `transaction_id: str`  
  - `user_id: str`
  - `amount: float`
  - `transaction_type: str`
  - `merchant_id: str`
  - `merchant_category: str`
  - `device_id: str`
  - `ip_address: str`
  - `location: str`
  - `created_at: datetime`

- **决策结果**：
  - `detection_method: str` → `"fast" | "llm"`
  - `fraud_score: float` → 最终风险分
  - `risk_level: str` → `"low" | "medium" | "high"`
  - `is_fraud: bool | null` → 事后标注（人工审核、逾期反馈等）

- **规则引擎相关**（模块一）：
  - `rule_decision: str` → `"pass" | "review" | "reject"`
  - `rule_reasons: text/json` → 字符串列表
  - `triggered_rules: text/json` → 规则命中详情：
    ```json
    [
      {"rule_code": "RULE_BIG_AMOUNT", "weight": 0.8},
      {"rule_code": "RULE_DEVICE_SHARE", "weight": 0.65}
    ]
    ```

- **LLM/Agent 相关**（模块二）：
  - `llm_decision: str` → `"accept" | "review" | "reject"`
  - `llm_risk_score: float`
  - `llm_confidence: float`
  - `llm_analysis: text`
  - `llm_reasoning: text`
  - `llm_agents_snapshot: text/json` → LangGraph 各 Agent 状态快照

> 其中 `llm_agents_snapshot` 是模块三 Phase 2 的关键字段，用于 LLM/Agent 审计。

### 4.2 后续可选统计表（Phase 2+）

> 非必须，可在性能或分析复杂度提需求时再落地。

- `model_daily_stats`：按模型+日期聚合的指标（TP/FP/TN/FN、AUC、Recall、FPR 等）
- `rule_stats`：按规则+日期聚合的触发次数、拒绝次数、命中欺诈数等

---

## 5. 子模块 A：实时监控 & 基础趋势（Phase 1 详细设计）

### 5.1 API 设计

#### A1. 实时总览

- **方法**: `GET /api/v1/analytics/realtime/summary`
- **参数**：
  - `window_minutes: int = 5` → 统计最近 N 分钟
- **返回示例**：
  ```json
  {
    "window": {
      "from": "2025-12-17T21:10:00Z",
      "to": "2025-12-17T21:15:00Z"
    },
    "traffic": {
      "total_requests": 1234,
      "fast_only": 1100,
      "llm_mode": 134
    },
    "decisions": {
      "pass": 1000,
      "review": 150,
      "reject": 84
    },
    "rates": {
      "reject_rate": 0.068,
      "review_rate": 0.122,
      "llm_trigger_rate": 0.108
    },
    "performance": {
      "fast_avg_latency_ms": 45.3,
      "llm_avg_latency_ms": 8123.7
    }
  }
  ```

#### A2. 按维度拆分统计

- **方法**: `GET /api/v1/analytics/realtime/by-dimension`
- **参数**：
  - `dim: str` → `"channel" | "region" | "detection_method" | "risk_level"`
  - `window_minutes: int = 60`
- **返回示例**：
  ```json
  {
    "dimension": "channel",
    "buckets": [
      {
        "key": "APP",
        "total_requests": 800,
        "reject_rate": 0.05,
        "llm_trigger_rate": 0.12
      },
      {
        "key": "WEB",
        "total_requests": 400,
        "reject_rate": 0.08,
        "llm_trigger_rate": 0.05
      }
    ]
  }
  ```

#### A3. 时间序列趋势

- **方法**: `GET /api/v1/analytics/realtime/timeseries`
- **参数**：
  - `metric: str` → `"requests" | "reject_rate" | "llm_trigger_rate" | "fast_latency" | "llm_latency"`
  - `granularity: str` → `"minute" | "5min" | "hour"`
  - `range_start`, `range_end`
- **返回示例**：
  ```json
  {
    "metric": "reject_rate",
    "granularity": "5min",
    "points": [
      {"ts": "2025-12-17T21:00:00Z", "value": 0.034},
      {"ts": "2025-12-17T21:05:00Z", "value": 0.071}
    ]
  }
  ```

### 5.2 实现要点

- 数据源：直接基于 `transactions` / `DetectionLog` 等表做聚合查询：
  - 按 `created_at` 过滤时间窗口
  - 按 `detection_method` 区分 fast / llm
- 初期不做物化统计表和缓存，优先保证功能可用；如访问量较大可再引入：
  - 按时间粒度的预聚合表
  - Redis 缓存最近窗口的结果

---

## 6. 子模块 B：模型 & 规则表现分析（Phase 2 详细设计）

### 6.1 模型表现接口（B1）

- **方法**: `GET /api/v1/analytics/model/performance`
- **参数**：
  - `model: str` → `"fast" | "llm" | "fusion"`
  - `range_start`, `range_end`
- **返回示例**：
  ```json
  {
    "model": "fast",
    "range": {"from": "2025-12-01", "to": "2025-12-17"},
    "summary": {
      "total": 100000,
      "fraud_labeled": 1200,
      "tp": 1100,
      "fp": 300,
      "tn": 97000,
      "fn": 100,
      "recall": 0.9167,
      "fpr": 0.0031,
      "precision": 0.7857,
      "auc": 0.92
    },
    "by_day": [
      {"date": "2025-12-10", "recall": 0.93, "fpr": 0.0028}
    ]
  }
  ```

> 说明：精确的 TP/FP/TN/FN 需要真值标注（`is_fraud`），在早期阶段可以允许为空，仅返回部分统计。

### 6.2 规则表现接口（B2）

- **方法**: `GET /api/v1/analytics/rules/stats`
- **返回示例**：
  ```json
  {
    "rules": [
      {
        "rule_code": "RULE_BIG_AMOUNT",
        "trigger_count": 523,
        "reject_count": 400,
        "review_count": 80,
        "hit_fraud_count": 350,
        "hit_normal_count": 50
      }
    ]
  }
  ```

### 6.3 与模块 1/2 的依赖

- 模块 1 需要确保 `_save_transaction` 或相关日志中**落地规则命中信息**：
  - `rule_decision`、`rule_reasons`、`triggered_rules`（JSON）
- 模块 2 的 LLM 决策（`llm_decision`、`llm_risk_score`）也写入 `transactions`，用于：
  - fast vs llm vs fusion 模型表现对比
  - 分析“被 LLM 纠正/覆盖”的样本集合

---

## 7. 子模块 C：LLM/Agent 决策审计（Phase 2 详细设计）

### 7.1 单笔决策链路接口（C1）

- **方法**: `GET /api/v1/analytics/llm/trace`
- **参数**：
  - `transaction_id: str`
- **返回示例**：
  ```json
  {
    "transaction_id": "test_api_001",
    "fast_result": {
      "fraud_score": 1.0,
      "risk_level": "high",
      "risk_factors": [
        "模型检测为高风险",
        "超大额交易 (权重: 0.8)"
      ]
    },
    "agents": {
      "coordinator": {...},
      "behavior": {...},
      "graph": {...},
      "rule": {...},
      "reflection": {...},
      "judge": {...}
    },
    "final_llm_output": {
      "llm_decision": "reject",
      "llm_risk_score": 0.98,
      "llm_confidence": 0.95,
      "llm_reasons": ["模型、行为、图、规则四维度一致判定高风险", "设备/IP 均处于高风险簇"],
      "llm_explanation": "...详细自然语言解释..."
    }
  }
  ```

### 7.2 Agent 统计接口（C2，可选）

- **方法**: `GET /api/v1/analytics/llm/agents/summary`
- **返回示例**：
  ```json
  {
    "range": {"from": "2025-12-10", "to": "2025-12-17"},
    "agents": {
      "behavior": {
        "high": 120,
        "medium": 300,
        "low": 580,
        "agree_with_judge_ratio": 0.82
      },
      "graph": {
        "high": 200,
        "medium": 400,
        "low": 400,
        "agree_with_judge_ratio": 0.91
      },
      "rule": {
        "high": 180,
        "medium": 500,
        "low": 320,
        "agree_with_judge_ratio": 0.88
      }
    }
  }
  ```

### 7.3 与模块 2 的对接点

- 在 `llm_service_v2.LlmAgentService` 的 JudgeAgent 阶段：
  - 将 LangGraph 各节点的输出（coordinator/behavior/graph/rule/reflection/judge）打包为 `llm_agents_snapshot` JSON
  - 随同 LLM 最终输出一起写入 `transactions` 表
- 模块三只需通过 `transaction_id` 查询 `transactions.llm_agents_snapshot` 并按上述格式返回，即可支撑审计页面。

---

## 8. 非功能要求与实施建议

### 8.1 非功能要求

- **性能**：
  - 模块三不在在线判决路径上，接口允许 100ms–500ms 级响应
  - 如后续访问压力增大，可引入预聚合表与缓存机制

- **安全与合规**：
  - 审计接口需避免泄露过多敏感信息（如完整卡号等），仅展示必要字段
  - 日志与审计数据按合规要求留痕、不可篡改

- **可扩展性**：
  - 接口以“读为主”的查询型 API 为主，易于新增维度（产品、渠道、活动等）

### 8.2 实施顺序建议

1. **Phase 1：实时监控**
   - 落地 `analytics_service` + `analytics` 路由
   - 实现 `realtime/summary`、`realtime/by-dimension`、`realtime/timeseries`
2. **扩展 `transactions` 字段**
   - 增加 `llm_decision`、`llm_risk_score`、`llm_agents_snapshot` 等字段
   - 修改模块二写入逻辑
3. **Phase 2：LLM 审计优先**
   - 实现 `GET /api/v1/analytics/llm/trace`
   - 验证单笔决策链路的还原能力
4. **Phase 2：模型 & 规则表现**
   - 实现 `model/performance` 与 `rules/stats` 接口
   - 结合 `is_fraud` 标注开展模型/规则评估

> 本设计文档可作为第三模块的统一说明，供后端、前端、数据与风控团队共同对齐需求与实现边界。
