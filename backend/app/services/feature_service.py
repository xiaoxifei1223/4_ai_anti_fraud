"""
特征工程服务 - 从交易数据中提取特征
"""
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.core.logger import logger
from app.models.schemas import FraudDetectionRequest
from app.models.models import Transaction, User
from app.services.graph_service import GraphFeatureService

# 初始化图特征服务
graph_service = GraphFeatureService()


def extract_features(request: FraudDetectionRequest, db: Session, use_graph: bool = True) -> np.ndarray:
    """
    从交易请求中提取完整特征
    
    特征组成:
    - 表格特征 (30维): 基础交易 + 用户历史 + 时间 + 设备/IP
    - 图特征 (15维): 节点特征 + 关系特征 + 子图特征
    - 总计: 45维特征向量
    
    Args:
        request: 欺诈检测请求
        db: 数据库会话
        use_graph: 是否启用图特征 (默认True)
    
    Returns:
        特征向量 (numpy array, 45维)
    """
    features = {}
    
    # === 表格特征 (30维) ===
    # 1. 基础交易特征 (2维)
    features.update(_extract_basic_features(request))
    
    # 2. 用户历史特征 (5维)
    features.update(_extract_user_history_features(request, db))
    
    # 3. 时间特征 (4维)
    features.update(_extract_time_features())
    
    # 4. 设备/IP特征 (2维)
    features.update(_extract_device_features(request, db))
    
    # 5. 扩展表格特征 (17维) - 保持30维表格特征总数
    features.update(_extract_extended_features(request, db))
    
    # 转换为numpy数组 - 表格特征部分
    table_features = np.array(list(features.values()), dtype=np.float32)
    
    # === 图特征 (15维) ===
    if use_graph:
        try:
            graph_features = graph_service.extract_graph_features(
                user_id=request.user_id,
                device_id=request.device_id,
                ip_address=request.ip_address,
                amount=request.amount
            )
            logger.debug(f"图特征提取成功 - 用户: {request.user_id}, 特征数: {len(graph_features)}")
        except Exception as e:
            logger.warning(f"图特征提取失败,使用零向量: {str(e)}")
            graph_features = np.zeros(15, dtype=np.float32)
    else:
        graph_features = np.zeros(15, dtype=np.float32)
    
    # === 特征融合 ===
    combined_features = np.concatenate([table_features, graph_features])
    
    logger.info(f"特征提取完成 - 交易ID: {request.transaction_id}, "
                f"表格特征: {len(table_features)}维, 图特征: {len(graph_features)}维, "
                f"总计: {len(combined_features)}维")
    
    return combined_features


def _extract_basic_features(request: FraudDetectionRequest) -> Dict[str, float]:
    """提取基础交易特征"""
    return {
        "amount": float(request.amount),
        "amount_log": np.log1p(float(request.amount)),
    }


def _extract_user_history_features(request: FraudDetectionRequest, db: Session) -> Dict[str, float]:
    """提取用户历史行为特征"""
    features = {}
    
    try:
        # 查询用户最近的交易
        recent_transactions = db.query(Transaction).filter(
            Transaction.transaction_time >= datetime.now() - timedelta(days=30)
        ).all()
        
        if recent_transactions:
            amounts = [t.amount for t in recent_transactions]
            features["user_transaction_count_30d"] = len(amounts)
            features["user_avg_amount_30d"] = np.mean(amounts)
            features["user_std_amount_30d"] = np.std(amounts) if len(amounts) > 1 else 0.0
            features["user_max_amount_30d"] = max(amounts)
            
            # 当前交易与历史的比值
            if features["user_avg_amount_30d"] > 0:
                features["amount_vs_avg_ratio"] = request.amount / features["user_avg_amount_30d"]
            else:
                features["amount_vs_avg_ratio"] = 0.0
        else:
            # 新用户或无历史交易
            features["user_transaction_count_30d"] = 0.0
            features["user_avg_amount_30d"] = 0.0
            features["user_std_amount_30d"] = 0.0
            features["user_max_amount_30d"] = 0.0
            features["amount_vs_avg_ratio"] = 0.0
    
    except Exception as e:
        logger.warning(f"提取用户历史特征失败: {str(e)}")
        features = {
            "user_transaction_count_30d": 0.0,
            "user_avg_amount_30d": 0.0,
            "user_std_amount_30d": 0.0,
            "user_max_amount_30d": 0.0,
            "amount_vs_avg_ratio": 0.0
        }
    
    return features


def _extract_time_features() -> Dict[str, float]:
    """提取时间特征"""
    now = datetime.now()
    
    return {
        "hour": float(now.hour),
        "day_of_week": float(now.weekday()),
        "is_weekend": 1.0 if now.weekday() >= 5 else 0.0,
        "is_night": 1.0 if now.hour < 6 or now.hour > 22 else 0.0,
    }


def _extract_device_features(request: FraudDetectionRequest, db: Session) -> Dict[str, float]:
    """提取设备和IP特征"""
    features = {}
    
    try:
        # 统计该设备的交易次数
        if request.device_id:
            device_count = db.query(Transaction).filter(
                Transaction.device_id == request.device_id,
                Transaction.transaction_time >= datetime.now() - timedelta(days=30)
            ).count()
            features["device_transaction_count"] = float(device_count)
        else:
            features["device_transaction_count"] = 0.0
        
        # 统计该IP的交易次数
        if request.ip_address:
            ip_count = db.query(Transaction).filter(
                Transaction.ip_address == request.ip_address,
                Transaction.transaction_time >= datetime.now() - timedelta(days=30)
            ).count()
            features["ip_transaction_count"] = float(ip_count)
        else:
            features["ip_transaction_count"] = 0.0
    
    except Exception as e:
        logger.warning(f"提取设备特征失败: {str(e)}")
        features = {
            "device_transaction_count": 0.0,
            "ip_transaction_count": 0.0
        }
    
    return features


def _extract_extended_features(request: FraudDetectionRequest, db: Session) -> Dict[str, float]:
    """
    提取扩展表格特征,补充到30维
    
    包括:
    - 交易金额分位数特征
    - 用户活跃度特征
    - 异常行为指标
    等
    """
    features = {}
    
    try:
        # 交易金额特征
        amount = float(request.amount)
        features["amount_sqrt"] = np.sqrt(amount)
        features["amount_square"] = amount ** 2
        features["amount_percentile"] = _calculate_amount_percentile(amount, db)
        
        # 用户活跃度特征 (7天)
        user_7d_count = db.query(Transaction).filter(
            Transaction.transaction_time >= datetime.now() - timedelta(days=7)
        ).count()
        features["user_transaction_count_7d"] = float(user_7d_count)
        
        # 短时高频特征 (1小时内)
        user_1h_count = db.query(Transaction).filter(
            Transaction.transaction_time >= datetime.now() - timedelta(hours=1)
        ).count()
        features["user_transaction_count_1h"] = float(user_1h_count)
        
        # 填充占位特征,确保达到30维表格特征
        for i in range(12):  # 补充12维占位特征
            features[f"extended_feature_{i+1}"] = 0.0
        
    except Exception as e:
        logger.warning(f"提取扩展特征失败: {str(e)}")
        # 默认值
        features["amount_sqrt"] = np.sqrt(float(request.amount))
        features["amount_square"] = float(request.amount) ** 2
        features["amount_percentile"] = 0.5
        features["user_transaction_count_7d"] = 0.0
        features["user_transaction_count_1h"] = 0.0
        for i in range(11):
            features[f"extended_feature_{i+1}"] = 0.0
    
    return features


def _calculate_amount_percentile(amount: float, db: Session) -> float:
    """
    计算当前交易金额在历史交易中的分位数
    """
    try:
        recent_amounts = db.query(Transaction.amount).filter(
            Transaction.transaction_time >= datetime.now() - timedelta(days=30)
        ).all()
        
        if recent_amounts:
            amounts_list = [a[0] for a in recent_amounts]
            percentile = np.searchsorted(sorted(amounts_list), amount) / len(amounts_list)
            return percentile
        else:
            return 0.5
    except:
        return 0.5


def get_feature_names() -> List[str]:
    """
    获取完整的特征名称列表 (45维)
    
    Returns:
        特征名称列表
    """
    # 表格特征名称 (30维)
    table_feature_names = [
        # 基础交易特征 (2维)
        "amount",
        "amount_log",
        
        # 用户历史特征 (5维)
        "user_transaction_count_30d",
        "user_avg_amount_30d",
        "user_std_amount_30d",
        "user_max_amount_30d",
        "amount_vs_avg_ratio",
        
        # 时间特征 (4维)
        "hour",
        "day_of_week",
        "is_weekend",
        "is_night",
        
        # 设备/IP特征 (2维)
        "device_transaction_count",
        "ip_transaction_count",
        
        # 扩展特征 (17维)
        "amount_sqrt",
        "amount_square",
        "amount_percentile",
        "user_transaction_count_7d",
        "user_transaction_count_1h",
    ]
    
    # 添加占位特征名称
    for i in range(12):
        table_feature_names.append(f"extended_feature_{i+1}")
    
    # 图特征名称 (15维)
    graph_feature_names = graph_service.get_feature_names()
    
    # 合并
    all_features = table_feature_names + graph_feature_names
    
    return all_features
