"""运行已训练策略进行团伙攻防模拟（骨架）

当前版本：
- 直接复用 train_group_attack_rl 的随机策略逻辑；
- 主要用于演示如何读取 SimulationGroupRun 以及扩展评估流程。
"""

from __future__ import annotations

import asyncio

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.core.logger import logger
from app.services.fraud_group_attack_env import FraudGroupAttackEnv, EnvConfig


async def main(num_episodes: int = 1) -> None:
    db: Session = SessionLocal()

    try:
        env = FraudGroupAttackEnv(db, env_config=EnvConfig())

        for i in range(num_episodes):
            state = env.reset()
            done = False
            step = 0
            while not done and step < env.env_config.max_steps:
                action = {"type": "ATTACK_TRANSACTION", "amount": 100.0}
                next_state, reward, done, info = await env.step(action)
                logger.info(f"step={step}, reward={reward}, decision={info['defense_result']['decision']}")
                state = next_state
                step += 1

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
