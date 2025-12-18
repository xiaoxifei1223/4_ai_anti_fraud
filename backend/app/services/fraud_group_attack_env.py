"""FraudGroupAttackEnv - 团伙 / 多账户协同攻防强化学习环境骨架

本环境遵循 Gym 风格接口设计，但不直接依赖任何外部 RL 库，
用于 Phase 1 中：
- 描述团伙攻击的状态（State）、动作（Action）和奖励（Reward）；
- 通过 DefenseAdapter 与现有防御系统交互；
- 为后续接入 PPO 等算法提供统一的交互接口。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.services.simulation_defense_adapter import DefenseAdapter, DefenseConfig


@dataclass
class EnvConfig:
    """环境配置（简化版, 包含阶段与资源约束配置）"""

    max_accounts: int = 50
    max_devices: int = 20
    max_ips: int = 30
    max_steps: int = 200
    initial_accounts: int = 5
    warmup_steps: int = 50


class FraudGroupAttackEnv:
    """团伙 / 多账户协同攻防环境骨架

    说明：
    - 为了降低实现复杂度，当前版本使用简化的内部状态与随机样本生成逻辑；
    - 后续可以逐步将状态空间替换为真实的图结构和行为统计特征。
    """

    def __init__(self, db: Session, env_config: Optional[EnvConfig] = None, defense_config: Optional[DefenseConfig] = None) -> None:
        self.db = db
        self.env_config = env_config or EnvConfig()
        self.defense_adapter = DefenseAdapter(db, defense_config)

        # 运行时变量
        self.current_step: int = 0
        self.num_accounts: int = 0
        # 团伙结构: 账户与设备/IP 关系
        self.accounts: Dict[str, Dict[str, Any]] = {}
        self.device_to_accounts: Dict[str, set] = {}
        self.ip_to_accounts: Dict[str, set] = {}
        # 行为记录: 每笔模拟交易
        self.transaction_history: List[Dict[str, Any]] = []

    def reset(self) -> Dict[str, Any]:
        """重置环境并返回初始状态"""

        self.current_step = 0
        self.num_accounts = 0
        self.accounts = {}
        self.device_to_accounts = {}
        self.ip_to_accounts = {}
        self.transaction_history = []

        # 初始化一批养号账户
        for _ in range(self.env_config.initial_accounts):
            self._register_account()

        state = self._build_state()
        logger.info(f"环境重置: {state}")
        return state

    async def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """执行一步环境交互

        Args:
            action: 高层动作描述（例如 {"type": "ATTACK_TRANSACTION", "amount": 1000.0}）。

        Returns:
            next_state, reward, done, info
        """

        self.current_step += 1

        action_type = (action.get("type") or "ATTACK_TRANSACTION").upper()
        phase = self._current_phase()

        defense_result: Optional[Dict[str, Any]] = None
        reward: float = 0.0

        # 1. 结构型动作: 注册账户 / 绑定资源
        if action_type == "REGISTER_ACCOUNT":
            created = self._register_account(action)
            # 注册成功有轻微成本, 失败则更高惩罚
            reward = -0.05 if created else -0.2
        elif action_type == "BIND_RESOURCE":
            success = self._bind_resource(action)
            reward = -0.02 if success else -0.1
        else:
            # 2. 交易型动作: 养号 / 爆雷 等
            tx = self._build_synthetic_transaction(action, phase)
            defense_results = await self.defense_adapter.evaluate_transactions([tx])
            defense_result = defense_results[0]

            decision = defense_result["decision"]
            if decision == "reject":
                reward = -1.0
            elif decision == "accept":
                reward = 1.0
            else:
                reward = -0.1  # review 视为轻微负向

            # 记录行为明细, 供后续 structure_summary / behavior_summary 使用
            self._record_transaction(tx, phase, action_type, decision)

        # 3. 构造下一个状态
        next_state = self._build_state()

        # 4. 终止条件：达到最大步数
        done = self.current_step >= self.env_config.max_steps

        info: Dict[str, Any] = {
            "defense_result": defense_result,
            "raw_action": action,
            "phase": phase,
            "action_type": action_type,
        }

        return next_state, reward, done, info

    def _build_synthetic_transaction(self, action: Dict[str, Any], phase: str) -> Dict[str, Any]:
        """根据高层动作构造一个合成交易样本

        为了不与现有特征工程 / 规则完全耦合，当前仅填充 FraudDetectionRequest
        所需的核心字段，并用随机数或动作参数填充。
        """

        action_type = (action.get("type") or "ATTACK_TRANSACTION").upper()

        # 选择账户: 优先使用 action 中指定的 account_id, 否则随机选择已有账户
        account_id = action.get("account_id")
        if not account_id or account_id not in self.accounts:
            if self.accounts:
                account_id = random.choice(list(self.accounts.keys()))
            else:
                account_id = self._register_account() or "acc-1"

        account_info = self.accounts[account_id]

        # 金额策略: 不同阶段 / 动作类型使用不同金额区间
        if action_type == "WARMUP_TRANSACTION" or phase == "warmup":
            base_amount = random.uniform(10.0, 200.0)
        elif action_type == "ATTACK_TRANSACTION":
            base_amount = random.uniform(1000.0, 8000.0)
        else:
            base_amount = random.uniform(50.0, 2000.0)

        amount = float(action.get("amount", base_amount))
        transaction_type = action.get("transaction_type", "purchase")

        device_id = action.get("device_id", account_info["device_id"])
        ip_address = action.get("ip_address", account_info["ip_address"])

        tx: Dict[str, Any] = {
            "transaction_id": action.get("transaction_id", f"sim-{self.current_step}-{account_id}"),
            "user_id": account_id,
            "amount": amount,
            "transaction_type": transaction_type,
            "merchant_id": action.get("merchant_id", "M123"),
            "merchant_category": action.get("merchant_category", "general"),
            "device_id": device_id,
            "ip_address": ip_address,
            "location": action.get("location", "CN"),
        }

        return tx

    # ===================== 结构与行为维护 =====================

    def _register_account(self, action: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """注册一个新账户, 并绑定设备/IP 资源"""

        if self.num_accounts >= self.env_config.max_accounts:
            return None

        idx = self.num_accounts + 1
        account_id = action.get("account_id") if action else None
        if not account_id:
            account_id = f"acc-{idx}"

        device_id = action.get("device_id") if action else None
        if not device_id:
            device_id = f"device-{random.randint(1, self.env_config.max_devices)}"

        ip_address = action.get("ip_address") if action else None
        if not ip_address:
            ip_address = f"10.0.0.{random.randint(1, self.env_config.max_ips)}"

        self.accounts[account_id] = {
            "device_id": device_id,
            "ip_address": ip_address,
        }

        self.device_to_accounts.setdefault(device_id, set()).add(account_id)
        self.ip_to_accounts.setdefault(ip_address, set()).add(account_id)

        self.num_accounts += 1
        return account_id

    def _bind_resource(self, action: Dict[str, Any]) -> bool:
        """调整账户与设备/IP 的绑定关系"""

        account_id = action.get("account_id")
        if not account_id or account_id not in self.accounts:
            return False

        account_info = self.accounts[account_id]
        updated = False

        new_device = action.get("device_id")
        if new_device and new_device != account_info["device_id"]:
            old_device = account_info["device_id"]
            if old_device in self.device_to_accounts:
                self.device_to_accounts[old_device].discard(account_id)
            self.device_to_accounts.setdefault(new_device, set()).add(account_id)
            account_info["device_id"] = new_device
            updated = True

        new_ip = action.get("ip_address")
        if new_ip and new_ip != account_info["ip_address"]:
            old_ip = account_info["ip_address"]
            if old_ip in self.ip_to_accounts:
                self.ip_to_accounts[old_ip].discard(account_id)
            self.ip_to_accounts.setdefault(new_ip, set()).add(account_id)
            account_info["ip_address"] = new_ip
            updated = True

        return updated

    def _record_transaction(self, tx: Dict[str, Any], phase: str, action_type: str, decision: str) -> None:
        """记录一次模拟交易行为"""

        self.transaction_history.append(
            {
                "step": self.current_step,
                "phase": phase,
                "action_type": action_type,
                "account_id": tx["user_id"],
                "device_id": tx["device_id"],
                "ip_address": tx["ip_address"],
                "amount": tx["amount"],
                "decision": decision,
            }
        )

    def _build_state(self) -> Dict[str, Any]:
        """构造当前状态, 提供给上层策略/LLM 使用"""

        phase = self._current_phase()
        num_accounts = len(self.accounts)
        num_devices = len(self.device_to_accounts)
        num_ips = len(self.ip_to_accounts)

        device_degrees = [len(v) for v in self.device_to_accounts.values()] or [0]
        ip_degrees = [len(v) for v in self.ip_to_accounts.values()] or [0]

        state: Dict[str, Any] = {
            "step": self.current_step,
            "phase": phase,
            "num_accounts": num_accounts,
            "num_devices": num_devices,
            "num_ips": num_ips,
            "device_account_histogram": self._compute_histogram(device_degrees),
            "ip_account_histogram": self._compute_histogram(ip_degrees),
        }
        return state

    def _current_phase(self) -> str:
        """根据当前步数判断处于养号期还是攻击期"""

        if self.current_step < self.env_config.warmup_steps:
            return "warmup"
        return "attack"

    @staticmethod
    def _compute_histogram(values: List[int]) -> Dict[str, int]:
        """将度分布统计为简单直方图"""

        buckets = {"1": 0, "2-3": 0, "4-5": 0, ">=6": 0}
        for v in values:
            if v <= 1:
                buckets["1"] += 1
            elif v <= 3:
                buckets["2-3"] += 1
            elif v <= 5:
                buckets["4-5"] += 1
            else:
                buckets[">=6"] += 1
        return buckets

    def collect_summaries(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """汇总本次运行的团伙结构与行为统计, 用于 SimulationGroupRun

        Returns:
            structure_summary, behavior_summary
        """

        # 结构统计
        device_degrees = [len(v) for v in self.device_to_accounts.values()]
        ip_degrees = [len(v) for v in self.ip_to_accounts.values()]

        structure_summary: Dict[str, Any] = {
            "device_account_histogram": self._compute_histogram(device_degrees or [0]),
            "ip_account_histogram": self._compute_histogram(ip_degrees or [0]),
            "max_accounts_per_device": max(device_degrees) if device_degrees else 0,
            "max_accounts_per_ip": max(ip_degrees) if ip_degrees else 0,
        }

        # 行为统计
        amount_bins = {"0-100": 0, "100-1000": 0, ">=1000": 0}
        phase_stats: Dict[str, Dict[str, float]] = {}
        decision_counts = {"accept": 0, "review": 0, "reject": 0}

        for rec in self.transaction_history:
            amt = float(rec["amount"])
            if amt < 100:
                amount_bins["0-100"] += 1
            elif amt < 1000:
                amount_bins["100-1000"] += 1
            else:
                amount_bins[">=1000"] += 1

            phase = rec["phase"]
            ps = phase_stats.setdefault(phase, {"transactions": 0, "total_amount": 0.0})
            ps["transactions"] += 1
            ps["total_amount"] += amt

            decision = rec["decision"]
            if decision in decision_counts:
                decision_counts[decision] += 1

        # 计算每个 phase 的平均金额
        phase_summary: Dict[str, Any] = {}
        for phase, ps in phase_stats.items():
            tx_count = ps["transactions"] or 1
            phase_summary[phase] = {
                "transactions": ps["transactions"],
                "avg_amount": ps["total_amount"] / tx_count,
            }

        behavior_summary: Dict[str, Any] = {
            "amount_distribution": amount_bins,
            "phase_stats": phase_summary,
            "decision_counts": decision_counts,
        }

        return structure_summary, behavior_summary
