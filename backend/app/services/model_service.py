"""
XGBoost 模型服务 - 模型加载和预测
"""
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from typing import Dict, Any
from app.core.logger import logger
from app.core.config import settings


class XGBoostModelService:
    """XGBoost 模型服务"""
    
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        self.model_loaded = False
        self.model_config = None
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            model_path = Path("models") / "xgboost_model.json"
            config_path = Path("models") / "model_config.json"
            
            if not model_path.exists():
                logger.warning(f"模型文件不存在: {model_path}")
                return False
            
            # 加载模型
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            logger.info(f"✅ XGBoost 模型加载成功: {model_path}")
            
            # 加载配置
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.model_config = json.load(f)
                    self.threshold = self.model_config.get('threshold', 0.5)
                    logger.info(f"✅ 模型配置加载成功，阈值: {self.threshold}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        使用模型进行预测
        
        Args:
            features: 特征向量 (45维)
            
        Returns:
            预测结果字典
        """
        if not self.model_loaded:
            raise RuntimeError("模型未加载")
        
        try:
            # 预测概率
            fraud_proba = self.model.predict_proba(features.reshape(1, -1))[0, 1]
            
            # 根据阈值判断
            is_fraud = fraud_proba >= self.threshold
            
            # 确定风险等级
            if fraud_proba >= 0.8:
                risk_level = "high"
            elif fraud_proba >= 0.5:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                'fraud_score': float(fraud_proba),
                'is_fraud': bool(is_fraud),
                'risk_level': risk_level,
                'threshold': self.threshold
            }
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model_loaded:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'threshold': self.threshold,
            'config': self.model_config
        }


# 全局模型服务实例
model_service = XGBoostModelService()
