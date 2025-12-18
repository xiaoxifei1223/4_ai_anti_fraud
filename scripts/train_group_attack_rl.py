"""训练团伙 / 多账户协同攻击 RL 策略（骨架）

当前版本：
- 不依赖具体 RL 库，仅演示如何与 FraudGroupAttackEnv 交互；
- 通过若干随机策略 Episode，生成基础的 SimulationGroupRun 记录。
"""

from __future__ import annotations

import asyncio
import uuid
import json
import random
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.core.logger import logger
from app.models.simulation_models import SimulationGroupRun
from app.services.fraud_group_attack_env import FraudGroupAttackEnv, EnvConfig


async def run_random_policy_episode(env: FraudGroupAttackEnv, max_steps: int) -> Dict[str, Any]:
    """使用高层随机策略跑一个 Episode，返回统计指标（占位实现）

    随机混合 REGISTER_ACCOUNT / WARMUP_TRANSACTION / ATTACK_TRANSACTION 等动作，
    以便在 structure_summary / behavior_summary 中体现差异化模式。
    """

    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps:
        phase = state.get("phase", "warmup")

        # 简单的阶段感知随机策略
        if phase == "warmup":
            # 养号期: 以注册新账号和小额交易为主
            if random.random() < 0.3:
                action = {"type": "REGISTER_ACCOUNT"}
            else:
                action = {"type": "WARMUP_TRANSACTION"}
        else:
            # 攻击期: 以高风险交易为主, 偶尔继续小额交易掩护
            r = random.random()
            if r < 0.1:
                action = {"type": "REGISTER_ACCOUNT"}
            elif r < 0.2:
                action = {"type": "WARMUP_TRANSACTION"}
            else:
                action = {"type": "ATTACK_TRANSACTION"}

        next_state, reward, done, info = await env.step(action)
        total_reward += reward
        steps += 1
        state = next_state

    return {
        "total_reward": total_reward,
        "steps": steps,
    }


async def main(num_episodes: int = 3) -> None:
    db: Session = SessionLocal()

    try:
        env = FraudGroupAttackEnv(db, env_config=EnvConfig())

        run_id = f"SIM-{uuid.uuid4().hex[:12]}"
        logger.info(f"启动随机策略模拟运行, run_id={run_id}")

        total_steps = 0
        total_reward = 0.0

        for i in range(num_episodes):
            stats = await run_random_policy_episode(env, max_steps=env.env_config.max_steps)
            logger.info(f"Episode {i+1}/{num_episodes} 结束: {stats}")
            total_steps += int(stats["steps"])
            total_reward += float(stats["total_reward"])

        avg_reward = total_reward / max(num_episodes, 1)

        # 从环境收集结构与行为摘要, 用于结构化审计
        structure_summary, behavior_summary = env.collect_summaries()

        # 写入 SimulationGroupRun（使用占位指标）
        sim_run = SimulationGroupRun(
            run_id=run_id,
            scenario_name="group_attack_random_baseline",
            description="随机策略基线模拟（仅用于验证环境与写库逻辑）",
            rl_algorithm="random-baseline",
            policy_version="v0",
            num_episodes=num_episodes,
            total_steps=total_steps,
            total_transactions=total_steps,  # 简化：每步一笔交易
            attack_success_rate=None,
            avg_reward_per_episode=avg_reward,
            avg_steps_to_success=None,
            defense_recall_drop=None,
            avg_accounts_used=None,
            structure_summary=json.dumps(structure_summary, ensure_ascii=False),
            behavior_summary=json.dumps(behavior_summary, ensure_ascii=False),
            status="finished",
        )

        db.add(sim_run)
        db.commit()
        logger.info(f"SimulationGroupRun 已写入数据库, id={sim_run.id}, run_id={sim_run.run_id}")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
