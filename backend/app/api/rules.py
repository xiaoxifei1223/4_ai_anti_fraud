"""
规则管理 API
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.schemas import RuleCreate, RuleResponse
from app.models.models import FraudRule
import uuid

router = APIRouter()


@router.post("/", response_model=RuleResponse)
async def create_rule(rule: RuleCreate, db: Session = Depends(get_db)):
    """创建新规则"""
    
    db_rule = FraudRule(
        rule_id=f"RULE_{uuid.uuid4().hex[:8].upper()}",
        rule_name=rule.rule_name,
        rule_type=rule.rule_type,
        description=rule.description,
        condition=rule.condition,
        threshold=rule.threshold,
        weight=rule.weight,
        priority=rule.priority,
        is_active=True,
    )
    
    db.add(db_rule)
    db.commit()
    db.refresh(db_rule)
    
    return db_rule


@router.get("/{rule_id}", response_model=RuleResponse)
async def get_rule(rule_id: str, db: Session = Depends(get_db)):
    """获取规则详情"""
    
    rule = db.query(FraudRule).filter(FraudRule.rule_id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    return rule


@router.get("/")
async def list_rules(
    is_active: bool = None,
    rule_type: str = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取规则列表"""
    
    query = db.query(FraudRule)
    
    if is_active is not None:
        query = query.filter(FraudRule.is_active == is_active)
    
    if rule_type:
        query = query.filter(FraudRule.rule_type == rule_type)
    
    rules = query.offset(skip).limit(limit).all()
    total = query.count()
    
    return {
        "total": total,
        "rules": rules,
        "skip": skip,
        "limit": limit
    }


@router.put("/{rule_id}/toggle")
async def toggle_rule(rule_id: str, db: Session = Depends(get_db)):
    """启用/禁用规则"""
    
    rule = db.query(FraudRule).filter(FraudRule.rule_id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    rule.is_active = not rule.is_active
    db.commit()
    
    return {"rule_id": rule_id, "is_active": rule.is_active}


@router.delete("/{rule_id}")
async def delete_rule(rule_id: str, db: Session = Depends(get_db)):
    """删除规则"""
    
    rule = db.query(FraudRule).filter(FraudRule.rule_id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    db.delete(rule)
    db.commit()
    
    return {"message": "规则已删除", "rule_id": rule_id}
