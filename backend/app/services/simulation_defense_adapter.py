"""DefenseAdapter 服务封装

用于模块 4 团伙 / 多账户协同攻防模拟场景下，将现有检测系统
（快速模块 + LLM 模块）抽象为统一的“防御黑盒”接口。

Phase 1 目标：
- 提供一个 evaluate_transactions 方法，支持批量评估一组交易请求；
- 统一输出决策结果（accept/review/reject）、分数、风险等级等；
- 预留对 fast_detect / llm_detect / full_detect 的集成入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.core.config import settings
from app.services import fraud_service


@dataclass
class DefenseConfig:
    """防御策略配置

    Phase 1 主要用于控制调用模式，后续可扩展模型版本、规则版本等。
    """

    mode: str = "fast+llm"  # "fast" | "llm" | "fast+llm"


class DefenseAdapter:
    """防御系统适配器

    在攻防模拟环境中，所有对防御系统的调用都通过该适配器完成，
    便于后续替换内部实现（例如切换为 HTTP 调用或多版本策略对比）。
    """

    def __init__(self, db: Session, config: Optional[DefenseConfig] = None) -> None:
        self.db = db
        self.config = config or DefenseConfig()

    async def evaluate_transactions(
        self,
        transactions: List[Dict[str, Any]],
        config_override: Optional[DefenseConfig] = None,
    ) -> List[Dict[str, Any]]:
        """批量评估一组交易请求

        Args:
            transactions: 交易请求列表，每个元素为与 FraudDetectionRequest 兼容的 dict。
            config_override: 可选的临时配置，若提供则优先于初始化时的配置。

        Returns:
            每笔交易对应一个统一结构的 dict：
            {
              "decision": "accept|review|reject",
              "fraud_score": float,
              "risk_level": "low|medium|high",
              "raw_mode": "fast|llm|full",
            }
        """

        cfg = config_override or self.config
        results: List[Dict[str, Any]] = []

        # 为简化 Phase 1 实现，暂时逐笔调用现有服务函数
        for tx in transactions:
            # 延迟导入 Pydantic Schema，避免在模型未就绪时阻塞整个服务
            try:
                from app.models.schemas import FraudDetectionRequest  # type: ignore
            except Exception as e:  # pragma: no cover - 仅在开发早期模型缺失时触发
                logger.error(f"导入 FraudDetectionRequest 失败，无法调用现有检测服务: {e}")
                raise

            request = FraudDetectionRequest(**tx)

            if cfg.mode == "fast":
                resp = await fraud_service.fast_detect(request, self.db)
                raw_mode = "fast"
            elif cfg.mode == "llm":
                resp = await fraud_service.llm_detect(request, self.db)
                raw_mode = "llm"
            else:
                # fast + llm 融合的完整流程
                resp = await fraud_service.full_detect(request, self.db)
                raw_mode = "full"

            decision = self._map_decision(resp.is_fraud, resp.risk_level)

            results.append(
                {
                    "decision": decision,
                    "fraud_score": float(resp.fraud_score),
                    "risk_level": resp.risk_level,
                    "raw_mode": raw_mode,
                }
            )

        return results

    @staticmethod
    def _map_decision(is_fraud: bool, risk_level: str) -> str:
        """将内部 is_fraud + risk_level 映射为 accept/review/reject 决策。

        - is_fraud=True 一律视为 reject；
        - is_fraud=False 且 risk_level=low 视为 accept；
        - 其他情况视为 review。
        """

        if is_fraud:
            return "reject"

        normalized = (risk_level or "").lower()
        if normalized == "low":
            return "accept"

        return "review"
