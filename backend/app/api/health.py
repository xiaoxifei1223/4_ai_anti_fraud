"""
健康检查和系统状态 API
"""
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.config import settings
from app.db.database import get_db
from app.models.schemas import HealthCheckResponse
from app.services.model_service import model_service

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    系统健康检查
    
    返回系统运行状态、数据库连接、模型加载等信息
    """
    
    # 检查数据库连接
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # 检查模型是否加载
    model_loaded = model_service.model_loaded
    
    # 确定整体状态
    status = "healthy" if db_status == "connected" else "unhealthy"
    
    return HealthCheckResponse(
        status=status,
        version=settings.VERSION,
        database=db_status,
        model_loaded=model_loaded,
        timestamp=datetime.now()
    )


@router.get("/ping")
async def ping():
    """简单的心跳检测"""
    return {"message": "pong", "timestamp": datetime.now()}
