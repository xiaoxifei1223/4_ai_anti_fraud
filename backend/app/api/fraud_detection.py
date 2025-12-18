"""
欺诈检测 API
"""
import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.logger import logger
from app.db.database import get_db
from app.models.schemas import FraudDetectionRequest, FraudDetectionResponse
from app.services import fraud_service

router = APIRouter()


@router.post("/detect", response_model=FraudDetectionResponse)
async def detect_fraud(
    request: FraudDetectionRequest,
    db: Session = Depends(get_db)
):
    """
    欺诈检测主接口
    
    支持三种检测模式：
    - fast: 快速检测（XGBoost + 规则引擎），毫秒级响应
    - llm: LLM深度分析，秒级响应  
    - full: 完整检测（快速检测 + LLM分析）
    """
    start_time = time.time()
    
    try:
        logger.info(f"收到欺诈检测请求 - 交易ID: {request.transaction_id}, 模式: {request.detection_mode}")
        
        # 根据检测模式调用不同的服务
        if request.detection_mode == "fast":
            result = await fraud_service.fast_detect(request, db)
        elif request.detection_mode == "llm":
            result = await fraud_service.llm_detect(request, db)
        elif request.detection_mode == "full":
            result = await fraud_service.full_detect(request, db)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的检测模式: {request.detection_mode}"
            )
        
        # 计算执行时间
        execution_time = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time
        
        logger.info(
            f"检测完成 - 交易ID: {request.transaction_id}, "
            f"欺诈分数: {result.fraud_score:.3f}, "
            f"耗时: {execution_time:.2f}ms"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"欺诈检测失败 - 交易ID: {request.transaction_id}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@router.post("/batch-detect")
async def batch_detect_fraud(
    requests: list[FraudDetectionRequest],
    db: Session = Depends(get_db)
):
    """
    批量欺诈检测
    
    支持同时检测多笔交易
    """
    results = []
    
    for request in requests:
        try:
            result = await detect_fraud(request, db)
            results.append(result)
        except Exception as e:
            logger.error(f"批量检测中失败 - 交易ID: {request.transaction_id}, 错误: {str(e)}")
            results.append({
                "transaction_id": request.transaction_id,
                "error": str(e)
            })
    
    return {"total": len(requests), "results": results}
