"""Analytics 服务层

提供模块三的统计与监控相关聚合查询逻辑。
当前阶段实现:
- Phase1: 实时总览接口所需的服务函数
- Phase2: C1 LLM 决策链路审计接口所需的查询函数
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.models import Transaction, DetectionLog


def get_realtime_summary(db: Session, window_minutes: int = 5) -> Dict[str, Any]:
    """获取最近一段时间内的实时概览统计。

    - 时间窗口基于 DetectionLog.created_at
    - 统计 fast / llm 两类检测的调用量和平均耗时
    - 交易层面统计基于 Transaction 表
    """
    if window_minutes <= 0:
        window_minutes = 5

    now = datetime.now()
    window_start = now - timedelta(minutes=window_minutes)

    # 1. 交易层面统计
    tx_query = db.query(Transaction).filter(Transaction.created_at >= window_start)
    total_requests = tx_query.count()

    fast_count = tx_query.filter(Transaction.detection_method == "fast").count()
    llm_count = tx_query.filter(Transaction.detection_method == "llm").count()

    pass_count = tx_query.filter(Transaction.risk_level == "low").count()
    review_count = tx_query.filter(Transaction.risk_level == "medium").count()
    reject_count = tx_query.filter(Transaction.risk_level == "high").count()

    # 2. 检测日志层面统计（性能指标）
    fast_perf = (
        db.query(func.avg(DetectionLog.execution_time_ms))
        .filter(
            DetectionLog.created_at >= window_start,
            DetectionLog.detection_type == "fast",
            DetectionLog.execution_time_ms.isnot(None),
        )
        .scalar()
    )

    llm_perf = (
        db.query(func.avg(DetectionLog.execution_time_ms))
        .filter(
            DetectionLog.created_at >= window_start,
            DetectionLog.detection_type == "llm",
            DetectionLog.execution_time_ms.isnot(None),
        )
        .scalar()
    )

    # 3. 组装结果
    traffic_fast_only = fast_count
    traffic_llm_mode = llm_count

    # 防止除零
    total_for_rate = total_requests or 1

    reject_rate = reject_count / total_for_rate
    review_rate = review_count / total_for_rate
    llm_trigger_rate = traffic_llm_mode / total_for_rate

    summary: Dict[str, Any] = {
        "window": {
            "from": window_start.isoformat(),
            "to": now.isoformat(),
            "minutes": window_minutes,
        },
        "traffic": {
            "total_requests": total_requests,
            "fast_only": traffic_fast_only,
            "llm_mode": traffic_llm_mode,
        },
        "decisions": {
            "pass": pass_count,
            "review": review_count,
            "reject": reject_count,
        },
        "rates": {
            "reject_rate": reject_rate,
            "review_rate": review_rate,
            "llm_trigger_rate": llm_trigger_rate,
        },
        "performance": {
            "fast_avg_latency_ms": fast_perf or 0.0,
            "llm_avg_latency_ms": llm_perf or 0.0,
        },
    }

    return summary


def get_llm_trace(db: Session, transaction_id: str) -> Optional[Dict[str, Any]]:
    """获取单笔交易的 LLM 决策链路审计信息。

    当前实现基于:
    - Transaction 表中的 fast / llm 交易记录
    - DetectionLog 表中 detection_type="llm" 的最新日志
    """
    # 获取 fast 检测结果
    fast_tx = (
        db.query(Transaction)
        .filter(
            Transaction.transaction_id == transaction_id,
            Transaction.detection_method == "fast",
        )
        .order_by(Transaction.created_at.desc())
        .first()
    )

    # 获取 LLM 检测日志
    llm_log = (
        db.query(DetectionLog)
        .join(Transaction, DetectionLog.transaction_id == Transaction.id)
        .filter(
            Transaction.transaction_id == transaction_id,
            DetectionLog.detection_type == "llm",
        )
        .order_by(DetectionLog.created_at.desc())
        .first()
    )

    if not fast_tx and not llm_log:
        return None

    # 解析 LLM Agents 快照
    agents_snapshot: Dict[str, Any] = {}
    if llm_log is not None and llm_log.llm_reasoning:
        try:
            parsed = json.loads(llm_log.llm_reasoning)
            if isinstance(parsed, dict):
                agents_snapshot = parsed
        except Exception:
            agents_snapshot = {}

    # 提取关键信息，字段命名与前端对齐
    coordinator = agents_snapshot.get("coordinator", {})
    behavior_agent = agents_snapshot.get("behavior", {})
    graph_agent = agents_snapshot.get("graph", {})
    rule_agent = agents_snapshot.get("rule", {})
    reflection = agents_snapshot.get("reflection", {})
    judge = agents_snapshot.get("judge", {})
    fast_result_snapshot = agents_snapshot.get("fast_result", {})

    # 组装 fast 模块结果
    fast_detection: Dict[str, Any] = {}
    if fast_tx is not None:
        fast_detection = {
            "fraudScore": fast_tx.fraud_score,
            "riskLevel": fast_tx.risk_level,
            "riskFactors": fast_result_snapshot.get("risk_factors", []),
            "detectedAt": fast_tx.created_at.isoformat() if fast_tx.created_at else None,
        }

    # 组装协调器决策
    coordinator_decision: Dict[str, Any] = {
        "executionMode": coordinator.get("execution_mode"),
        "agentsToRun": coordinator.get("agents_to_run", []),
        "fastScore": coordinator.get("fast_score"),
        "reason": coordinator.get("reason"),
    }

    # 组装各 Agent 分析结果
    behavior_analysis: Dict[str, Any] = {
        "riskLevel": behavior_agent.get("behavior_risk_level"),
        "reasons": behavior_agent.get("behavior_reasons", []),
        "thoughts": behavior_agent.get("thoughts", []),
        "toolCalls": behavior_agent.get("tool_calls", []),
        "skipped": behavior_agent.get("skipped", False),
        "error": behavior_agent.get("error"),
    }

    graph_analysis: Dict[str, Any] = {
        "riskLevel": graph_agent.get("graph_risk_level"),
        "reasons": graph_agent.get("graph_reasons", []),
        "thoughts": graph_agent.get("thoughts", []),
        "toolCalls": graph_agent.get("tool_calls", []),
        "skipped": graph_agent.get("skipped", False),
        "error": graph_agent.get("error"),
    }

    rule_analysis: Dict[str, Any] = {
        "riskLevel": rule_agent.get("rule_risk_level"),
        "reasons": rule_agent.get("rule_reasons", []),
        "thoughts": rule_agent.get("thoughts", []),
        "toolCalls": rule_agent.get("tool_calls", []),
        "skipped": rule_agent.get("skipped", False),
        "error": rule_agent.get("error"),
    }

    # 组装反思验证结果
    reflection_result: Dict[str, Any] = {
        "isConsistent": reflection.get("is_consistent"),
        "concerns": reflection.get("concerns", []),
        "recommendation": reflection.get("recommendation"),
    }

    # 组装最终裁决
    final_decision: Dict[str, Any] = {
        "decision": judge.get("llm_decision"),
        "riskScore": judge.get("llm_risk_score"),
        "confidence": judge.get("llm_confidence"),
        "reasons": judge.get("llm_reasons", []),
        "explanation": judge.get("llm_explanation") or (llm_log.llm_analysis if llm_log else None),
    }

    # 添加元数据
    metadata: Dict[str, Any] = {
        "modelVersion": llm_log.model_version if llm_log else None,
        "detectionType": "llm",
        "analyzedAt": llm_log.created_at.isoformat() if llm_log and llm_log.created_at else None,
    }

    # 最终返回结构（适合前端展示）
    trace: Dict[str, Any] = {
        "transactionId": transaction_id,
        "metadata": metadata,
        "fastDetection": fast_detection or None,
        "coordinatorDecision": coordinator_decision,
        "agentAnalysis": {
            "behavior": behavior_analysis,
            "graph": graph_analysis,
            "rule": rule_analysis,
        },
        "reflection": reflection_result,
        "finalDecision": final_decision,
    }

    return trace
