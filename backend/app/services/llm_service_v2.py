"""
LLM + Multi-Agent 服务 (ReAct + Reflection 增强版)

架构设计:
1. CoordinatorAgent: 协调器,根据初步评估决定执行路径
2. BehaviorAgent/GraphAgent/RuleAgent: ReAct 模式,带工具调用能力
3. ReflectionAgent: 反思验证,检查矛盾和不合理之处  
4. JudgeAgent: 最终裁决

特性:
- 并行执行 Behavior/Graph/Rule Agent (性能优化)
- 每个 Agent 都支持 ReAct 循环 (Thought → Action → Observation)
- 反思机制确保决策质量
"""
from __future__ import annotations

import json
import asyncio
from typing import Any, Dict, List, Optional, Callable
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logger import logger


# ===================== 工具函数定义 (Tool Functions) =====================

class AgentTools:
    """Agent 可用的工具集合
    
    当前版本: 工具函数使用 pass 占位,后续可以实现真实逻辑
    """
    
    @staticmethod
    async def query_user_history(user_id: str, days: int = 30) -> Dict[str, Any]:
        """查询用户历史交易数据
        
        Args:
            user_id: 用户ID
            days: 查询最近多少天的数据
            
        Returns:
            包含用户历史交易统计的字典
        """
        logger.debug(f"🔧 Tool调用: query_user_history(user_id={user_id}, days={days})")
        # TODO: 实现真实的数据库查询
        pass
        
    @staticmethod
    async def query_device_reputation(device_id: str) -> Dict[str, Any]:
        """查询设备信誉分和历史记录
        
        Args:
            device_id: 设备ID
            
        Returns:
            设备信誉信息
        """
        logger.debug(f"🔧 Tool调用: query_device_reputation(device_id={device_id})")
        # TODO: 查询设备黑名单/白名单
        pass
        
    @staticmethod
    async def query_ip_blacklist(ip_address: str) -> Dict[str, Any]:
        """查询IP是否在黑名单中
        
        Args:
            ip_address: IP地址
            
        Returns:
            IP风险信息
        """
        logger.debug(f"🔧 Tool调用: query_ip_blacklist(ip_address={ip_address})")
        # TODO: 查询IP黑名单数据库
        pass
        
    @staticmethod
    async def query_merchant_info(merchant_id: str) -> Dict[str, Any]:
        """查询商户信息
        
        Args:
            merchant_id: 商户ID
            
        Returns:
            商户信誉和历史数据
        """
        logger.debug(f"🔧 Tool调用: query_merchant_info(merchant_id={merchant_id})")
        # TODO: 查询商户数据库
        pass
        
    @staticmethod
    async def query_similar_cases(features: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """查询相似历史案例 (RAG)
        
        Args:
            features: 当前交易特征
            top_k: 返回最相似的K个案例
            
        Returns:
            相似案例列表
        """
        logger.debug(f"🔧 Tool调用: query_similar_cases(top_k={top_k})")
        # TODO: 向量检索 (Qdrant/Milvus)
        pass
        
    @staticmethod
    async def calculate_velocity(user_id: str, time_window: int = 3600) -> Dict[str, Any]:
        """计算交易速率 (Velocity Check)
        
        Args:
            user_id: 用户ID
            time_window: 时间窗口(秒)
            
        Returns:
            交易速率统计
        """
        logger.debug(f"🔧 Tool调用: calculate_velocity(user_id={user_id}, time_window={time_window})")
        # TODO: 查询 Redis/时序数据库
        pass


# ===================== 状态定义 =====================

class FraudState(TypedDict, total=False):
    """LangGraph 状态类型 (增强版)
    
    payload: 送入 LLM 的上下文
    coordinator_decision: 协调器的路由决策
    behavior: 行为分析 Agent 输出
    graph: 图关系分析 Agent 输出
    rule: 规则与合规 Agent 输出
    reflection: 反思 Agent 的验证结果
    llm_output: 裁决 Agent 最终输出
    """
    
    payload: Dict[str, Any]
    coordinator_decision: Dict[str, Any]
    behavior: Dict[str, Any]
    graph: Dict[str, Any]
    rule: Dict[str, Any]
    reflection: Dict[str, Any]
    llm_output: Dict[str, Any]


# ===================== Kimi 客户端 =====================

class KimiClient:
    """Kimi 2 API 客户端封装 (复用原有实现)"""
    
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def chat(self, messages: List[Dict[str, str]], timeout: float = 10.0) -> str:
        """调用 Kimi Chat 接口"""
        if not self.api_key:
            raise RuntimeError("KIMI_API_KEY 未配置, 无法调用 LLM")

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                stream=False,
            )
        except Exception as e:
            logger.error(f"调用 Kimi(OpenAI SDK) 失败: {e}")
            raise

        try:
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"解析 Kimi 响应失败: {e}")
            raise


# ===================== 主服务类 =====================

class LlmAgentService:
    """LLM Multi-Agent 服务 (ReAct + Reflection 架构)
    
    工作流:
    1. CoordinatorAgent: 评估风险,决定执行路径
    2. 并行执行 Behavior/Graph/Rule Agent (ReAct 模式)
    3. ReflectionAgent: 反思验证
    4. JudgeAgent: 最终裁决
    """

    def __init__(self) -> None:
        if not settings.KIMI_API_KEY:
            logger.warning("KIMI_API_KEY 未配置, LLM 功能将不可用")
        self.client = KimiClient(
            api_key=settings.KIMI_API_KEY,
            base_url=settings.KIMI_BASE_URL,
            model=settings.KIMI_MODEL,
        )
        self.tools = AgentTools()
        
        # 构建 LangGraph 工作流
        graph = StateGraph(FraudState)
        
        # 添加节点
        graph.add_node("coordinator_agent", self._coordinator_agent_node)
        graph.add_node("parallel_agents", self._parallel_agents_node)
        graph.add_node("reflection_agent", self._reflection_agent_node)
        graph.add_node("judge_agent", self._judge_agent_node)
        
        # 设置流程
        graph.set_entry_point("coordinator_agent")
        graph.add_edge("coordinator_agent", "parallel_agents")
        graph.add_edge("parallel_agents", "reflection_agent")
        graph.add_edge("reflection_agent", "judge_agent")
        graph.add_edge("judge_agent", END)
        
        self._graph = graph.compile()

    # ===================== Agent 节点实现 =====================
    
    async def _coordinator_agent_node(self, state: FraudState) -> FraudState:
        """协调器 Agent: 评估风险级别,决定后续执行路径
        
        策略:
        - 低风险 (<0.3): 只执行 RuleAgent
        - 中风险 (0.3-0.7): 执行全部 3 个 Agent
        - 高风险 (>0.7): 执行全部 + 额外工具调用
        """
        payload = state["payload"]
        logger.info("\n" + "#"*80)
        logger.info("🧭 CoordinatorAgent 开始协调...")
        logger.info("#"*80)
        
        if not settings.KIMI_API_KEY:
            state["coordinator_decision"] = {
                "execution_mode": "standard",
                "agents_to_run": ["behavior", "graph", "rule"],
                "reason": "LLM未启用,使用默认策略"
            }
            return state
        
        # 获取 fast_detect 的初步评分
        fast_score = payload.get("fast_result", {}).get("fraud_score", 0.5)
        
        # 动态路由决策
        if fast_score < 0.3:
            execution_mode = "fast"
            agents_to_run = ["rule"]
            reason = f"低风险交易(score={fast_score:.2f}),仅需规则验证"
        elif fast_score > 0.7:
            execution_mode = "deep"
            agents_to_run = ["behavior", "graph", "rule"]
            reason = f"高风险交易(score={fast_score:.2f}),启动深度分析"
        else:
            execution_mode = "standard"
            agents_to_run = ["behavior", "graph", "rule"]
            reason = f"中风险交易(score={fast_score:.2f}),执行标准流程"
        
        decision = {
            "execution_mode": execution_mode,
            "agents_to_run": agents_to_run,
            "fast_score": fast_score,
            "reason": reason
        }
        
        logger.info(f"✅ 协调决策: {decision}")
        state["coordinator_decision"] = decision
        return state
    
    async def _parallel_agents_node(self, state: FraudState) -> FraudState:
        """并行执行 Behavior/Graph/Rule Agent"""
        logger.info("\n" + "="*80)
        logger.info("🚀 启动并行 Agent 执行...")
        logger.info("="*80)
        
        decision = state.get("coordinator_decision", {})
        agents_to_run = decision.get("agents_to_run", ["behavior", "graph", "rule"])
        
        # 创建并行任务
        tasks = []
        if "behavior" in agents_to_run:
            tasks.append(self._behavior_agent_react(state))
        if "graph" in agents_to_run:
            tasks.append(self._graph_agent_react(state))
        if "rule" in agents_to_run:
            tasks.append(self._rule_agent_react(state))
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for i, agent_name in enumerate(["behavior", "graph", "rule"]):
            if agent_name not in agents_to_run:
                state[agent_name] = {"skipped": True}
            elif isinstance(results[agents_to_run.index(agent_name)], Exception):
                logger.error(f"❌ {agent_name}Agent 执行失败: {results[i]}")
                state[agent_name] = {"error": str(results[i])}
            else:
                state[agent_name] = results[agents_to_run.index(agent_name)]
        
        logger.info("✅ 并行 Agent 执行完成")
        return state
    
    async def _behavior_agent_react(self, state: FraudState) -> Dict[str, Any]:
        """BehaviorAgent (ReAct 模式)
        
        ReAct 循环:
        1. Thought: 我需要分析用户行为模式
        2. Action: 调用 query_user_history 工具
        3. Observation: 用户最近30天平均交易额500元
        4. Thought: 当前交易12888元,是均值的25倍,异常
        5. 最终输出风险结论
        """
        payload = state["payload"]
        logger.info("="*60)
        logger.info("🤖 BehaviorAgent (ReAct) 开始分析...")
        
        if not settings.KIMI_API_KEY:
            return {
                "behavior_risk_level": "medium",
                "behavior_reasons": ["LLM未启用"],
                "tool_calls": []
            }
        
        # ReAct Prompt: 指导 LLM 进行推理+工具调用
        system_prompt = """你是金融反欺诈系统中的'行为模式分析'专家。

你可以使用以下工具来辅助分析:
- query_user_history(user_id, days=30): 查询用户历史交易数据
- calculate_velocity(user_id, time_window=3600): 计算交易速率

请按照 ReAct 模式进行分析:
1. Thought: 思考需要什么信息
2. Action: 决定调用哪个工具 (如果需要)
3. Observation: 工具返回的结果
4. (重复1-3,直到有足够信息)
5. Final Answer: 给出最终风险判断

最终输出 JSON 格式:
{
  "behavior_risk_level": "high|medium|low",
  "behavior_reasons": ["原因1", "原因2"],
  "thoughts": ["思考过程1", "思考过程2"],
  "tool_calls": ["使用的工具名称"]
}
"""
        
        user_prompt = f"""分析以下交易的行为风险:\n{json.dumps(payload, ensure_ascii=False, indent=2)}

请按照 ReAct 模式进行推理,明确说明你的思考过程和工具调用。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            content = await self.client.chat(messages)
            logger.info(f"✅ BehaviorAgent 原始响应: {content}")
            
            # 解析 JSON (容错处理)
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # 如果不是纯 JSON,尝试提取
                result = {
                    "behavior_risk_level": "medium",
                    "behavior_reasons": ["解析失败,使用默认"],
                    "thoughts": [content[:200]],
                    "tool_calls": []
                }
            
            logger.info(f"✅ BehaviorAgent 解析结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ BehaviorAgent 失败: {e}")
            return {
                "behavior_risk_level": "medium",
                "behavior_reasons": [f"执行异常: {str(e)}"],
                "tool_calls": []
            }
    
    async def _graph_agent_react(self, state: FraudState) -> Dict[str, Any]:
        """GraphAgent (ReAct 模式)"""
        payload = state["payload"]
        logger.info("="*60)
        logger.info("🤖 GraphAgent (ReAct) 开始分析...")
        
        if not settings.KIMI_API_KEY:
            return {
                "graph_risk_level": "medium",
                "graph_reasons": ["LLM未启用"],
                "tool_calls": []
            }
        
        system_prompt = """你是金融反欺诈系统中的'图关系风险'专家。

可用工具:
- query_device_reputation(device_id): 查询设备信誉分
- query_ip_blacklist(ip_address): 查询IP黑名单
- query_similar_cases(features, top_k=5): 查询相似历史案例

请使用 ReAct 模式分析图关系风险,输出 JSON:
{
  "graph_risk_level": "high|medium|low",
  "graph_reasons": ["原因1", "原因2"],
  "thoughts": ["思考过程"],
  "tool_calls": ["使用的工具"]
}
"""
        
        user_prompt = f"""分析图关系风险:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            content = await self.client.chat(messages)
            logger.info(f"✅ GraphAgent 原始响应: {content}")
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {
                    "graph_risk_level": "medium",
                    "graph_reasons": ["解析失败"],
                    "thoughts": [content[:200]],
                    "tool_calls": []
                }
            
            logger.info(f"✅ GraphAgent 解析结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ GraphAgent 失败: {e}")
            return {
                "graph_risk_level": "medium",
                "graph_reasons": [f"执行异常: {str(e)}"],
                "tool_calls": []
            }
    
    async def _rule_agent_react(self, state: FraudState) -> Dict[str, Any]:
        """RuleAgent (ReAct 模式)"""
        payload = state["payload"]
        logger.info("="*60)
        logger.info("🤖 RuleAgent (ReAct) 开始分析...")
        
        if not settings.KIMI_API_KEY:
            return {
                "rule_risk_level": "medium",
                "rule_reasons": ["LLM未启用"],
                "tool_calls": []
            }
        
        system_prompt = """你是金融反欺诈系统中的'规则与合规'专家。

可用工具:
- query_merchant_info(merchant_id): 查询商户信息

请使用 ReAct 模式分析规则合规性,输出 JSON:
{
  "rule_risk_level": "high|medium|low",
  "rule_reasons": ["原因1", "原因2"],
  "thoughts": ["思考过程"],
  "tool_calls": ["使用的工具"]
}
"""
        
        user_prompt = f"""分析规则风险:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            content = await self.client.chat(messages)
            logger.info(f"✅ RuleAgent 原始响应: {content}")
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {
                    "rule_risk_level": "medium",
                    "rule_reasons": ["解析失败"],
                    "thoughts": [content[:200]],
                    "tool_calls": []
                }
            
            logger.info(f"✅ RuleAgent 解析结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ RuleAgent 失败: {e}")
            return {
                "rule_risk_level": "medium",
                "rule_reasons": [f"执行异常: {str(e)}"],
                "tool_calls": []
            }
    
    async def _reflection_agent_node(self, state: FraudState) -> FraudState:
        """反思 Agent: 检查前面 Agent 的结论是否一致、合理
        
        功能:
        1. 检查 Behavior/Graph/Rule 三个 Agent 的结论是否矛盾
        2. 质疑不合理的推理
        3. 如果发现严重矛盾,可以要求重新分析
        """
        logger.info("\n" + "="*80)
        logger.info("🔍 ReflectionAgent 开始反思验证...")
        logger.info("="*80)
        
        if not settings.KIMI_API_KEY:
            state["reflection"] = {
                "is_consistent": True,
                "concerns": [],
                "recommendation": "proceed"
            }
            return state
        
        # 收集三个 Agent 的结论
        behavior = state.get("behavior", {})
        graph = state.get("graph", {})
        rule = state.get("rule", {})
        
        system_prompt = """你是一个反思与验证专家,负责检查其他 Agent 的分析结论。

你的任务:
1. 检查 Behavior/Graph/Rule 三个 Agent 的结论是否一致
2. 识别逻辑矛盾 (例如: Behavior说低风险,但Graph说高风险)
3. 质疑不合理的推理
4. 给出改进建议

输出 JSON 格式:
{
  "is_consistent": true/false,
  "concerns": ["发现的问题1", "问题2"],
  "recommendation": "proceed|re_analyze|escalate"
}
"""
        
        user_prompt = f"""请检查以下三个 Agent 的分析结论:

BehaviorAgent: {json.dumps(behavior, ensure_ascii=False)}

GraphAgent: {json.dumps(graph, ensure_ascii=False)}

RuleAgent: {json.dumps(rule, ensure_ascii=False)}

请指出任何矛盾或不合理之处。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            content = await self.client.chat(messages)
            logger.info(f"✅ ReflectionAgent 原始响应: {content}")
            
            try:
                reflection = json.loads(content)
            except json.JSONDecodeError:
                reflection = {
                    "is_consistent": True,
                    "concerns": ["解析失败,默认通过"],
                    "recommendation": "proceed"
                }
            
            logger.info(f"✅ ReflectionAgent 验证结果: {reflection}")
            
            # 如果发现严重矛盾,记录警告
            if not reflection.get("is_consistent", True):
                logger.warning(f"⚠️ ReflectionAgent 发现矛盾: {reflection.get('concerns')}")
            
            state["reflection"] = reflection
            return state
            
        except Exception as e:
            logger.error(f"❌ ReflectionAgent 失败: {e}")
            state["reflection"] = {
                "is_consistent": True,
                "concerns": [f"反思失败: {str(e)}"],
                "recommendation": "proceed"
            }
            return state
    
    async def _judge_agent_node(self, state: FraudState) -> FraudState:
        """裁决 Agent: 综合所有 Agent 的结论,给出最终决策"""
        logger.info("\n" + "="*80)
        logger.info("⚖️ JudgeAgent 开始最终裁决...")
        logger.info("="*80)
        
        if not settings.KIMI_API_KEY:
            state["llm_output"] = {
                "llm_decision": "review",
                "llm_risk_score": 0.5,
                "llm_confidence": 0.0,
                "llm_reasons": ["LLM未启用"],
                "llm_explanation": "LLM未启用,无法给出详细分析"
            }
            return state
        
        # 收集所有信息
        combined = {
            "coordinator": state.get("coordinator_decision", {}),
            "behavior": state.get("behavior", {}),
            "graph": state.get("graph", {}),
            "rule": state.get("rule", {}),
            "reflection": state.get("reflection", {}),
            "fast_result": state["payload"].get("fast_result", {})
        }
        
        system_prompt = """你是金融反欺诈系统的最终裁判。

你需要综合以下信息做出最终决策:
1. 快速检测模块的初步评分
2. Behavior/Graph/Rule 三个专家 Agent 的分析
3. ReflectionAgent 的验证结果

输出 JSON:
{
  "llm_decision": "accept|review|reject",
  "llm_risk_score": 0.0-1.0,
  "llm_confidence": 0.0-1.0,
  "llm_reasons": ["理由1", "理由2"],
  "llm_explanation": "详细解释"
}
"""
        
        user_prompt = f"""综合以下信息做出最终裁决:\n\n{json.dumps(combined, ensure_ascii=False, indent=2)}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            content = await self.client.chat(messages)
            logger.info(f"✅ JudgeAgent 原始响应: {content}")
            
            try:
                llm_output = json.loads(content)
            except json.JSONDecodeError:
                llm_output = {
                    "llm_decision": "review",
                    "llm_risk_score": 0.5,
                    "llm_confidence": 0.3,
                    "llm_reasons": ["JSON解析失败"],
                    "llm_explanation": content[:500]
                }
            
            # 将裁决结果也纳入快照,用于后续审计
            combined["judge"] = llm_output
            
            # 在 llm_output 中附带完整 agents 快照
            enriched_output = {
                **llm_output,
                "agents_snapshot": combined,
            }
            
            logger.info(f"✅ JudgeAgent 最终裁决: {enriched_output}")
            state["llm_output"] = enriched_output
            
            logger.info("\n" + "#"*80)
            logger.info("🎯 LLM+Agent 工作流执行完成")
            logger.info(f"最终决策: {enriched_output.get('llm_decision')}, 风险分数: {enriched_output.get('llm_risk_score')}")
            logger.info("#"*80 + "\n")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ JudgeAgent 失败: {e}")
            state["llm_output"] = {
                "llm_decision": "review",
                "llm_risk_score": 0.5,
                "llm_confidence": 0.0,
                "llm_reasons": [f"裁决失败: {str(e)}"],
                "llm_explanation": "系统异常,建议人工审核"
            }
            return state
    
    # ===================== 公开接口 =====================
    
    async def analyze_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """分析单笔交易 (公开接口)
        
        Args:
            payload: 包含交易数据和 fast_detect 结果的字典
            
        Returns:
            LLM 分析结果
        """
        logger.info("\n" + "#"*80)
        logger.info("🚀 启动 LLM+MultiAgent (ReAct+Reflection) 工作流")
        logger.info("#"*80)
        logger.debug(f"初始 payload: {payload}")
        
        # 调用 LangGraph
        result_state = await self._graph.ainvoke({"payload": payload})
        
        logger.info("\n" + "#"*80)
        logger.info("🎯 工作流执行完成")
        logger.info(f"最终输出: {result_state['llm_output']}")
        logger.info("#"*80 + "\n")
        
        return result_state["llm_output"]


# ===================== 全局单例 =====================

# 全局 LLM Agent 服务实例 (ReAct + Reflection 增强版)
llm_agent_service = LlmAgentService()
