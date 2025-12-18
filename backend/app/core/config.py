"""
系统配置模块
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载项目根目录下的 .env 环境变量
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(DOTENV_PATH)


class Settings(BaseSettings):
    """系统配置"""
    
    # 项目信息
    PROJECT_NAME: str = "反欺诈系统"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # 数据库配置 - SQLite
    DATABASE_URL: str = "sqlite:///./data/fraud_detection.db"
    
    # Neo4j 配置
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # Redis 配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # LLM 配置 (Kimi 2)
    KIMI_API_KEY: str = os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    KIMI_BASE_URL: str = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    KIMI_MODEL: str = os.getenv("KIMI_MODEL", "moonshot-v1-128k")
    
    # 模型配置
    MODEL_PATH: Path = Path("models")
    XGBOOST_MODEL_PATH: Path = MODEL_PATH / "xgboost_model.json"
    
    # 特征工程配置
    FEATURE_CACHE_TTL: int = 300  # 特征缓存时间（秒）
    
    # 业务配置
    FAST_FRAUD_THRESHOLD: float = 0.7  # 快速检测阈值
    LLM_FRAUD_THRESHOLD: float = 0.6   # LLM检测阈值
    MAX_TRANSACTION_AMOUNT: float = 1000000.0  # 最大交易金额
    
    # 性能指标
    TARGET_RECALL_RATE: float = 0.95  # 目标召回率
    TARGET_FPR: float = 0.05  # 目标误报率
    
    # 日志配置
    LOG_LEVEL: str = "DEBUG"  # 调试阶段使用 DEBUG 级别,查看详细日志
    LOG_FILE: str = "logs/fraud_detection.log"
    
    class Config:
        env_file = DOTENV_PATH
        case_sensitive = True
        extra = "ignore"


# 创建全局配置实例
settings = Settings()
