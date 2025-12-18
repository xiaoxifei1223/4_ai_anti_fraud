"""Analytics 相关 API

提供模块三（Analytics）对外的 HTTP 接口。
当前阶段实现:
- Phase1: 实时总览接口
- Phase2: C1 LLM 决策链路审计接口
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.analytics_service import get_realtime_summary, get_llm_trace

router = APIRouter()


@router.get("/analytics/realtime/summary")
async def realtime_summary(window_minutes: int = 5, db: Session = Depends(get_db)):
    """获取最近一段时间的实时概览统计。

    - `window_minutes`: 时间窗口（分钟），默认 5 分钟
    """
    summary = get_realtime_summary(db, window_minutes=window_minutes)
    return summary


@router.get("/analytics/llm/trace")
async def llm_trace(transaction_id: str = Query(..., description="交易ID"), db: Session = Depends(get_db)):
    """获取单笔交易的 LLM 决策链路审计信息。"""
    trace = get_llm_trace(db, transaction_id=transaction_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"未找到交易 {transaction_id} 的 LLM 审计记录")
    return trace
