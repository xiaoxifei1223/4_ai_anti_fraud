"""
日志配置模块
"""
import sys
from pathlib import Path
from loguru import logger
from app.core.config import settings

# 移除默认处理器
logger.remove()

# 创建日志目录
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 添加控制台输出
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
    colorize=True,
)

# 添加文件输出
logger.add(
    settings.LOG_FILE,
    rotation="500 MB",
    retention="30 days",
    encoding="utf-8",
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

__all__ = ["logger"]
