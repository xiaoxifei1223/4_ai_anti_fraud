"""
LLM + Agent 服务封装

当前阶段目标:
- 提供统一的 Kimi2 客户端封装
- 提供一个基础的 LLM 分析接口, 后续可平滑迁移到 LangGraph 多 Agent 工作流
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict

import httpx
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logger import logger


class FraudState(TypedDict, total=False):
    """LangGraph 状态类型

    payload: 送入 LLM 的上下文
    behavior: 行为分析 Agent 输出
    graph: 图关系分析 Agent 输出
    rule: 规则与合规 Agent 输出
    llm_output: 裁决 Agent 最终输出
    """

    payload: Dict[str, Any]
    behavior: Dict[str, Any]
    graph: Dict[str, Any]
    rule: Dict[str, Any]
    llm_output: Dict[str, Any]


class KimiClient:
    """Kimi 2 API 客户端封装

    使用 OpenAI 兼容 SDK 调用 Kimi 的 Chat Completions 接口。
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        # 使用官方 AsyncOpenAI 客户端, 兼容 Moonshot/Kimi 接口
        self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def chat(self, messages: List[Dict[str, str]], timeout: float = 10.0) -> str:
        """调用 Kimi Chat 接口, 返回 assistant 的文本内容。

        Args:
            messages: OpenAI 风格的对话列表, 每项包含 role/content
            timeout: 请求超时时间秒 (目前由 HTTP 客户端控制, OpenAI SDK 内置)
        """
        if not self.api_key:
            raise RuntimeError("KIMI_API_KEY 未配置, 无法调用 LLM")

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                stream=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"调用 Kimi(OpenAI SDK) 失败: {e}")
            raise

        try:
            # 新版 SDK 返回对象, 直接从 choices 读取内容
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            logger.error(f"解析 Kimi 响应失败: {e}; 原始响应对象: {resp}")
            raise


class LlmAgentService:
    """LLM Agent 服务

    基于 LangGraph 的多 Agent 工作流:
    - BehaviorAgent: 行为模式分析
    - GraphAgent: 图关系风险分析
    - RuleAgent: 规则与合规分析
    - JudgeAgent: 综合裁决与解释
    """

    def __init__(self) -> None:
        if not settings.KIMI_API_KEY:
            logger.warning("KIMI_API_KEY 未配置, LLM 功能将不可用")
        self.client = KimiClient(
            api_key=settings.KIMI_API_KEY,
            base_url=settings.KIMI_BASE_URL,
            model=settings.KIMI_MODEL,
        )

        # 构建 LangGraph: 多 Agent 串联工作流
        graph = StateGraph(FraudState)
        graph.add_node("behavior_agent", self._behavior_agent_node)
        graph.add_node("graph_agent", self._graph_agent_node)
        graph.add_node("rule_agent", self._rule_agent_node)
        graph.add_node("judge_agent", self._judge_agent_node)

        graph.set_entry_point("behavior_agent")
        graph.add_edge("behavior_agent", "graph_agent")
        graph.add_edge("graph_agent", "rule_agent")
        graph.add_edge("rule_agent", "judge_agent")
        graph.add_edge("judge_agent", END)

        self._graph = graph.compile()

    async def _behavior_agent_node(self, state: FraudState) -> FraudState:
        """行为模式分析 Agent 节点"""
        payload = state["payload"]
        logger.info("="*60)
        logger.info("🤖 BehaviorAgent 开始分析...")
        logger.debug(f"输入 payload: {payload}")
    
        if not settings.KIMI_API_KEY:
            state["behavior"] = {
                "behavior_risk_level": "medium",
                "behavior_reasons": ["LLM 未启用, 使用默认行为分析结果"],
            }
            logger.warning("BehaviorAgent: KIMI_API_KEY 未配置, 使用默认结果")
            return state
    
        system_prompt = (
            "你是金融反欺诈系统中的'行为模式分析'专家, "
            "负责从用户行为和交易模式角度评估风险。"
        )
        user_prompt = (
            "下面是与当前交易相关的 JSON 数据:\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "请只从'用户行为和交易模式'的角度进行分析, 给出行为风险等级和简要理由。\n"
            "严格按照以下 JSON 格式输出, 不要包含多余文字:\n"
            "{\n"
            '  "behavior_risk_level": "high|medium|low",\n'
            '  "behavior_reasons": ["原因1", "原因2"]\n'
            "}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
        logger.info(f"BehaviorAgent Prompt (user): {user_prompt[:200]}...")
            
        try:
            content = await self.client.chat(messages)
            logger.info(f"✅ BehaviorAgent 原始响应: {content}")
                
            parsed = json.loads(content)
            behavior = {
                "behavior_risk_level": str(parsed.get("behavior_risk_level", "medium")),
                "behavior_reasons": parsed.get("behavior_reasons", []) or [],
            }
            logger.info(f"✅ BehaviorAgent 解析结果: {behavior}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ BehaviorAgent 调用或解析失败: {e}")
            behavior = {
                "behavior_risk_level": "medium",
                "behavior_reasons": ["BehaviorAgent 输出异常, 使用默认结果"],
            }
            logger.warning(f"BehaviorAgent 使用兜底结果: {behavior}")
    
        state["behavior"] = behavior
        logger.info("="*60)
        return state
    
    async def _graph_agent_node(self, state: FraudState) -> FraudState:
        """图关系风险 Agent 节点"""
        payload = state["payload"]
        logger.info("="*60)
        logger.info("🤖 GraphAgent 开始分析...")
        logger.debug(f"输入 payload: {payload}")
    
        if not settings.KIMI_API_KEY:
            state["graph"] = {
                "graph_risk_level": "medium",
                "graph_reasons": ["LLM 未启用, 使用默认图关系分析结果"],
            }
            logger.warning("GraphAgent: KIMI_API_KEY 未配置, 使用默认结果")
            return state
    
        system_prompt = (
            "你是金融反欺诈系统中的'图关系风险'专家, "
            "负责从设备/IP/地址共享度、社区欺诈率等角度评估风险。"
        )
        user_prompt = (
            "下面是与当前交易相关的 JSON 数据(包含模型初判和风险因素):\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "请只从'图关系与关联网络'的角度进行分析, 给出图关系风险等级和简要理由。\n"
            "严格按照以下 JSON 格式输出, 不要包含多余文字:\n"
            "{\n"
            '  "graph_risk_level": "high|medium|low",\n'
            '  "graph_reasons": ["原因1", "原因2"]\n'
            "}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
        logger.info(f"GraphAgent Prompt (user): {user_prompt[:200]}...")
            
        try:
            content = await self.client.chat(messages)
            logger.info(f"✅ GraphAgent 原始响应: {content}")
                
            parsed = json.loads(content)
            graph_result = {
                "graph_risk_level": str(parsed.get("graph_risk_level", "medium")),
                "graph_reasons": parsed.get("graph_reasons", []) or [],
            }
            logger.info(f"✅ GraphAgent 解析结果: {graph_result}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ GraphAgent 调用或解析失败: {e}")
            graph_result = {
                "graph_risk_level": "medium",
                "graph_reasons": ["GraphAgent 输出异常, 使用默认结果"],
            }
            logger.warning(f"GraphAgent 使用兜底结果: {graph_result}")
    
        state["graph"] = graph_result
        logger.info("="*60)
        return state
    
    async def _rule_agent_node(self, state: FraudState) -> FraudState:
        """规则与合规 Agent 节点"""
        payload = state["payload"]
        logger.info("="*60)
        logger.info("🤖 RuleAgent 开始分析...")
        logger.debug(f"输入 payload: {payload}")
    
        if not settings.KIMI_API_KEY:
            state["rule"] = {
                "rule_risk_level": "medium",
                "rule_reasons": ["LLM 未启用, 使用默认规则分析结果"],
            }
            logger.warning("RuleAgent: KIMI_API_KEY 未配置, 使用默认结果")
            return state
    
        system_prompt = (
            "你是金融反欺诈系统中的'规则与合规'专家, "
            "负责基于命中规则和业务策略评估是否需要拒绝或人工审核。"
        )
        user_prompt = (
            "下面是与当前交易相关的 JSON 数据(包含命中规则和风险因素):\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "请只从'规则与合规'的角度进行分析, 给出规则风险等级和简要理由。\n"
            "严格按照以下 JSON 格式输出, 不要包含多余文字:\n"
            "{\n"
            '  "rule_risk_level": "high|medium|low",\n'
            '  "rule_reasons": ["原因1", "原因2"]\n'
            "}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
        logger.info(f"RuleAgent Prompt (user): {user_prompt[:200]}...")
            
        try:
            content = await self.client.chat(messages)
            logger.info(f"✅ RuleAgent 原始响应: {content}")
                
            parsed = json.loads(content)
            rule_result = {
                "rule_risk_level": str(parsed.get("rule_risk_level", "medium")),
                "rule_reasons": parsed.get("rule_reasons", []) or [],
            }
            logger.info(f"✅ RuleAgent 解析结果: {rule_result}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ RuleAgent 调用或解析失败: {e}")
            rule_result = {
                "rule_risk_level": "medium",
                "rule_reasons": ["RuleAgent 输出异常, 使用默认结果"],
            }
            logger.warning(f"RuleAgent 使用兜底结果: {rule_result}")
    
        state["rule"] = rule_result
        logger.info("="*60)
        return state
    
    async def _judge_agent_node(self, state: FraudState) -> FraudState:
        """裁决 Agent 节点: 综合各维度结果给出最终决策"""
        logger.info("="*60)
        logger.info("🤖 JudgeAgent 开始综合裁决...")
        logger.info(f"已有分析结果 - Behavior: {state.get('behavior')}")
        logger.info(f"已有分析结果 - Graph: {state.get('graph')}")
        logger.info(f"已有分析结果 - Rule: {state.get('rule')}")
        
        combined_payload: Dict[str, Any] = {
            "request": state["payload"].get("request", {}),
            "fast_result": state["payload"].get("fast_result", {}),
            "behavior": state.get("behavior") or {},
            "graph": state.get("graph") or {},
            "rule": state.get("rule") or {},
        }
        
        logger.debug(f"JudgeAgent 综合输入: {combined_payload}")

        llm_output = await self._analyze_transaction_core(combined_payload)
        state["llm_output"] = llm_output
        
        logger.info(f"✅ JudgeAgent 最终决策: {llm_output}")
        logger.info("="*60)
        return state

    async def _analyze_transaction_core(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """底层 LLM 调用 + 解析逻辑, 被 LangGraph 节点复用"""
        logger.info("JudgeAgent 准备调用 LLM 进行综合裁决...")
        
        if not self.client.api_key:
            # 返回一个安全的默认结果, 避免线上直接报错
            logger.warning("KIMI_API_KEY 未配置, 使用默认 LLM 分析结果")
            return {
                "llm_decision": "review",
                "llm_risk_score": 0.5,
                "llm_confidence": 0.0,
                "llm_reasons": ["LLM 未启用, 使用默认兜底结果"],
                "llm_explanation": "由于未配置 LLM 接入, 系统将该笔交易标记为需要人工审核。",
            }

        system_prompt = (
            "你是一名金融反欺诈风控专家, 需要综合模型初判结果、行为分析、图关系分析和规则分析, "
            "给出这笔交易的最终风险评估和决策建议。"
        )

        user_prompt = (
            "请根据以下 JSON 数据, 综合各个维度的分析结果, 评估该笔交易是否存在欺诈风险, "
            "并严格按照指定 JSON 格式输出:\n\n"
            "{payload}\n\n"
            "输出 JSON 格式如下(不要包含多余文字):\n"
            "{{\n"
            "  \"llm_decision\": \"accept|review|reject\",\n"
            "  \"llm_risk_score\": 0.0-1.0 之间的小数,\n"
            "  \"llm_confidence\": 0.0-1.0 之间的小数,\n"
            "  \"llm_reasons\": [\"原因1\", \"原因2\"],\n"
            "  \"llm_explanation\": \"面向风控/审核员的自然语言解释\"\n"
            "}}"
        ).format(payload=json.dumps(payload, ensure_ascii=False))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"JudgeAgent Prompt (user): {user_prompt[:300]}...")
        
        try:
            content = await self.client.chat(messages)
            logger.info(f"✅ JudgeAgent 原始响应: {content}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ 调用 Kimi LLM 失败: {e}")
            return {
                "llm_decision": "review",
                "llm_risk_score": 0.5,
                "llm_confidence": 0.0,
                "llm_reasons": ["LLM 调用失败, 使用默认兜底结果"],
                "llm_explanation": "由于 LLM 调用失败, 建议人工审核本笔交易。",
            }

        # 解析 LLM 返回的 JSON
        try:
            parsed = json.loads(content)
            llm_decision = str(parsed.get("llm_decision", "review"))
            llm_risk_score = float(parsed.get("llm_risk_score", 0.5))
            llm_confidence = float(parsed.get("llm_confidence", 0.0))
            llm_reasons = parsed.get("llm_reasons", []) or []
            llm_explanation = parsed.get("llm_explanation", "")
            
            logger.info(f"✅ JudgeAgent 解析结果 - decision: {llm_decision}, score: {llm_risk_score}, confidence: {llm_confidence}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ 解析 LLM JSON 输出失败: {e}; content={content}")
            llm_decision = "review"
            llm_risk_score = 0.5
            llm_confidence = 0.0
            llm_reasons = ["LLM 输出无法解析, 使用默认兜底结果"]
            llm_explanation = "由于 LLM 输出格式异常, 建议人工审核本笔交易。"

        return {
            "llm_decision": llm_decision,
            "llm_risk_score": llm_risk_score,
            "llm_confidence": llm_confidence,
            "llm_reasons": llm_reasons,
            "llm_explanation": llm_explanation,
        }

    async def analyze_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """针对单笔交易做 LLM 深度分析。

        当前通过 LangGraph 状态机调用, 后续可扩展为多 Agent。

        payload 建议包含:
        - request: 原始 FraudDetectionRequest 的 dict
        - fast_result: fast_detect 的结果 dict (可选)
        """
        logger.info("\n" + "#"*80)
        logger.info("🚀 开始 LLM+Agent 多维度分析工作流")
        logger.info("#"*80)
        logger.debug(f"初始 payload: {payload}")
        
        # 使用 LangGraph 的异步调用
        result_state = await self._graph.ainvoke({"payload": payload})
        
        logger.info("\n" + "#"*80)
        logger.info("🎯 LLM+Agent 工作流执行完成")
        logger.info(f"最终输出: {result_state['llm_output']}")
        logger.info("#"*80 + "\n")
        
        return result_state["llm_output"]


# 全局 LLM Agent 服务实例
llm_agent_service = LlmAgentService()
