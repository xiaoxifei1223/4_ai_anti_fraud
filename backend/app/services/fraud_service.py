"""
欺诈检测服务 - 核心业务逻辑
"""
from datetime import datetime
from typing import List
import json
from sqlalchemy.orm import Session
from app.core.logger import logger
from app.core.config import settings
from app.models.schemas import FraudDetectionRequest, FraudDetectionResponse
from app.models.models import Transaction, DetectionLog
from app.services.model_service import model_service
from app.services.feature_service import extract_features
from app.services.rule_service import RuleEngine
from app.services.llm_service_v2 import llm_agent_service  # 使用增强版 (ReAct + Reflection)


async def fast_detect(
    request: FraudDetectionRequest,
    db: Session
) -> FraudDetectionResponse:
    """
    快速检测模式 - XGBoost + 规则引擎
    目标：毫秒级响应
    """
    logger.debug(f"执行快速检测 - 交易ID: {request.transaction_id}")
    
    risk_factors = []
    rule_decision = 'pass'
    rule_reasons = []
    
    # 1. 提取特征
    try:
        features = extract_features(request, db)
    except Exception as e:
        logger.warning(f"特征提取失败，使用简化特征: {str(e)}")
        # 如果特征提取失败，使用基础特征（仅金额）
        import numpy as np
        features = np.zeros(45, dtype=np.float32)
        features[0] = float(request.amount)  # 第1维: 金额
    
    # 2. XGBoost 模型预测
    if model_service.model_loaded:
        try:
            prediction = model_service.predict(features)
            fraud_score = prediction['fraud_score']
            is_fraud_ml = prediction['is_fraud']
            risk_level = prediction['risk_level']
            
            if is_fraud_ml:
                risk_factors.append("模型检测为高风险")
        except Exception as e:
            logger.error(f"XGBoost 预测失败: {str(e)}")
            fraud_score = 0.3
            risk_level = "low"
    else:
        logger.warning("模型未加载，使用规则判断")
        fraud_score = 0.3
        risk_level = "low"
    
    # 3. 规则引擎检查
    try:
        rule_engine = RuleEngine(db)
        rule_result = rule_engine.evaluate(request)
        
        if rule_result['triggered']:
            # 规则被触发
            rule_score = rule_result['risk_score']
            rule_decision = rule_result['decision']
            rule_reasons = rule_result['reasons']
            
            # 将规则分数融合到最终分数
            fraud_score = max(fraud_score, rule_score)  # 取模型和规则的最大值
            
            # 添加规则风险因素
            for rule_info in rule_result['triggered_rules']:
                risk_factors.append(f"{rule_info['rule_name']} (\u6743\u91cd: {rule_info['weight']})")
            
            logger.info(f"触发 {len(rule_result['triggered_rules'])} 条规则，决策: {rule_decision}")
    except Exception as e:
        logger.error(f"规则引擎检查失败: {str(e)}")
        rule_decision = 'pass'
    
    # 限制分数范围
    fraud_score = min(fraud_score, 1.0)
    
    # 判断是否欺诈 - 优先使用规则决策
    if rule_decision == 'reject':
        is_fraud = True
    elif rule_decision == 'review':
        is_fraud = fraud_score >= 0.5  # review 情况下降低阈值
    else:
        is_fraud = fraud_score >= settings.FAST_FRAUD_THRESHOLD
    
    # 确定风险等级
    if fraud_score >= 0.8 or rule_decision == 'reject':
        risk_level = "high"
    elif fraud_score >= 0.5 or rule_decision == 'review':
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # 生成建议
    if rule_decision == 'reject':
        recommendation = f"建议拒绝交易: {', '.join(rule_reasons)}"
    elif is_fraud:
        recommendation = "建议拒绝交易并进行人工审核"
    elif risk_level == "medium":
        recommendation = f"建议进行额外验证（短信验证码等）: {', '.join(rule_reasons) if rule_reasons else '模型检测为中等风险'}"
    else:
        recommendation = "可以放行"
    
    # 保存交易记录
    _save_transaction(request, fraud_score, is_fraud, risk_level, "fast", db)
    
    return FraudDetectionResponse(
        transaction_id=request.transaction_id,
        is_fraud=is_fraud,
        fraud_score=fraud_score,
        risk_level=risk_level,
        risk_factors=risk_factors,
        detection_method="fast",
        execution_time_ms=0.0,  # 会在API层计算
        recommendation=recommendation
    )


async def llm_detect(
    request: FraudDetectionRequest,
    db: Session
) -> FraudDetectionResponse:
    """
    LLM检测模式 - 深度分析
    目标：秒级响应
    """
    logger.debug(f"执行LLM检测 - 交易ID: {request.transaction_id}")
    
    # 1. 先执行快速检测, 复用 fast_detect 的逻辑
    fast_result = await fast_detect(request, db)
    
    # 2. 构造传给 LLM 的上下文 payload
    payload = {
        "request": {
            "transaction_id": request.transaction_id,
            "user_id": request.user_id,
            "amount": float(request.amount),
            "transaction_type": request.transaction_type,
            "merchant_id": request.merchant_id,
            "merchant_category": request.merchant_category,
            "device_id": request.device_id,
            "ip_address": request.ip_address,
            "location": request.location,
        },
        "fast_result": {
            "fraud_score": fast_result.fraud_score,
            "risk_level": fast_result.risk_level,
            "risk_factors": fast_result.risk_factors,
        },
    }
    
    # 3. 调用 LLM Agent 服务
    llm_output = await llm_agent_service.analyze_transaction(payload)
    llm_decision = llm_output.get("llm_decision", "review")
    llm_risk_score = float(llm_output.get("llm_risk_score", fast_result.fraud_score))
    llm_confidence = float(llm_output.get("llm_confidence", 0.0))
    llm_reasons = [str(r) for r in llm_output.get("llm_reasons", [])]
    llm_explanation = llm_output.get("llm_explanation", "")
    
    # 4. 将 LLM 风险分数与 fast 模块结果做一个简单融合
    fraud_score = (fast_result.fraud_score + llm_risk_score) / 2.0
    
    # 根据融合后的分数判断风险等级
    if fraud_score >= 0.8:
        risk_level = "high"
    elif fraud_score >= 0.5:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # LLM 决策优先影响是否需要人工审核
    if llm_decision == "reject":
        is_fraud = True
    elif llm_decision == "accept":
        is_fraud = False
    else:
        is_fraud = fraud_score >= settings.LLM_FRAUD_THRESHOLD
    
    risk_factors = fast_result.risk_factors + llm_reasons
    
    if llm_decision == "reject":
        recommendation = "LLM 建议拒绝交易并进行人工审核"
    elif llm_decision == "accept" and risk_level == "low":
        recommendation = "LLM 分析认为风险可控, 可以放行"
    else:
        recommendation = "LLM 建议人工复核本笔交易"
    
    # 保存交易记录
    transaction = _save_transaction(request, fraud_score, is_fraud, risk_level, "llm", db)
    
    # 记录 LLM 检测日志,包含多 Agent 决策快照,用于审计
    agents_snapshot = llm_output.get("agents_snapshot")
    try:
        detection_log = DetectionLog(
            transaction_id=transaction.id,
            detection_type="llm",
            model_version="llm-service-v2",
            fraud_probability=llm_risk_score,
            prediction=is_fraud,
            risk_factors=llm_reasons,
            features=None,
            graph_features=None,
            triggered_rules=None,
            llm_analysis=llm_explanation,
            llm_reasoning=json.dumps(agents_snapshot, ensure_ascii=False) if agents_snapshot is not None else None,
            execution_time_ms=None,
        )
        db.add(detection_log)
        db.commit()
    except Exception as e:
        logger.error(f"保存 LLM 检测日志失败 - 交易ID: {request.transaction_id}, 错误: {str(e)}")
        db.rollback()
    
    return FraudDetectionResponse(
        transaction_id=request.transaction_id,
        is_fraud=is_fraud,
        fraud_score=fraud_score,
        risk_level=risk_level,
        risk_factors=risk_factors,
        detection_method="llm",
        execution_time_ms=0.0,
        recommendation=recommendation,
        llm_analysis=llm_explanation,
        llm_reasoning=llm_explanation,
    )


async def full_detect(
    request: FraudDetectionRequest,
    db: Session
) -> FraudDetectionResponse:
    """
    完整检测模式 - 快速检测 + LLM分析
    综合两种模式的结果
    """
    logger.debug(f"执行完整检测 - 交易ID: {request.transaction_id}")
    
    # 先执行快速检测
    fast_result = await fast_detect(request, db)
    
    # 如果快速检测已经判定为欺诈，直接返回
    if fast_result.is_fraud:
        fast_result.detection_method = "full"
        return fast_result
    
    # 如果快速检测风险中等，调用LLM深度分析
    if fast_result.risk_level == "medium":
        llm_result = await llm_detect(request, db)
        
        # 综合两个结果
        final_score = (fast_result.fraud_score + llm_result.fraud_score) / 2
        final_risk_factors = list(set(fast_result.risk_factors + llm_result.risk_factors))
        
        is_fraud = final_score >= settings.FAST_FRAUD_THRESHOLD
        
        if final_score >= 0.8:
            risk_level = "high"
        elif final_score >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return FraudDetectionResponse(
            transaction_id=request.transaction_id,
            is_fraud=is_fraud,
            fraud_score=final_score,
            risk_level=risk_level,
            risk_factors=final_risk_factors,
            detection_method="full",
            execution_time_ms=0.0,
            recommendation=llm_result.recommendation,
            llm_analysis=llm_result.llm_analysis,
            llm_reasoning=llm_result.llm_reasoning
        )
    
    # 低风险直接返回快速检测结果
    fast_result.detection_method = "full"
    return fast_result


def _save_transaction(
    request: FraudDetectionRequest,
    fraud_score: float,
    is_fraud: bool,
    risk_level: str,
    detection_method: str,
    db: Session,
) -> Transaction:
    """保存交易记录到数据库"""
    try:
        transaction = Transaction(
            transaction_id=request.transaction_id,
            amount=request.amount,
            transaction_type=request.transaction_type,
            merchant_id=request.merchant_id,
            merchant_category=request.merchant_category,
            device_id=request.device_id,
            ip_address=request.ip_address,
            location=request.location,
            transaction_time=datetime.now(),
            is_fraud=is_fraud,
            fraud_score=fraud_score,
            risk_level=risk_level,
            detection_method=detection_method,
            review_status="pending"
        )
        
        db.add(transaction)
        db.commit()
        db.refresh(transaction)
        
        logger.debug(f"交易记录已保存 - 交易ID: {request.transaction_id}")
        return transaction
    except Exception as e:
        logger.error(f"保存交易记录失败 - 交易ID: {request.transaction_id}, 错误: {str(e)}")
        db.rollback()
        raise
