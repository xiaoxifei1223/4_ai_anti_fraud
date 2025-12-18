"""
规则引擎服务 - 基于规则的欺诈检测
"""
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.logger import logger
from app.models.schemas import FraudDetectionRequest
from app.models.models import FraudRule, Transaction


class RuleEngine:
    """规则引擎"""
    
    def __init__(self, db: Session, config_path: str = "config/fraud_rules.yaml"):
        self.db = db
        self.config_path = Path(config_path)
        self.rules_config = self._load_yaml_config()
        self.rule_strategy = self.rules_config.get('rule_strategy', {})
        self.active_rules = self._parse_yaml_rules()
        
        logger.info(f"从 YAML 加载了 {len(self.active_rules)} 条激活规则")
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """从 YAML 文件加载规则配置"""
        try:
            if not self.config_path.exists():
                logger.warning(f"规则配置文件不存在: {self.config_path}")
                return {'rules': [], 'rule_strategy': {}}
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"成功加载规则配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载 YAML 配置失败: {e}")
            return {'rules': [], 'rule_strategy': {}}
    
    def _parse_yaml_rules(self) -> List[Dict[str, Any]]:
        """解析 YAML 规则为内部格式"""
        rules = []
        for rule_config in self.rules_config.get('rules', []):
            if rule_config.get('is_active', True):
                rules.append(rule_config)
        
        # 按优先级排序
        rules.sort(key=lambda x: x.get('priority', 0), reverse=True)
        return rules
    
    def evaluate(self, request: FraudDetectionRequest) -> Dict[str, Any]:
        """
        评估交易是否触发规则
        
        Returns:
            {
                'triggered': bool,
                'triggered_rules': List[str],
                'risk_score': float,
                'decision': str,  # 'pass', 'review', 'reject'
                'reasons': List[str]
            }
        """
        triggered_rules = []
        total_weight = 0.0
        high_priority_count = 0
        medium_priority_count = 0
        reasons = []
        
        # 遍历所有规则
        for rule in self.active_rules:
            if self._check_rule_conditions(rule, request):
                rule_info = {
                    'rule_id': rule.get('rule_id'),
                    'rule_name': rule.get('rule_name'),
                    'rule_type': rule.get('rule_type'),
                    'weight': rule.get('weight', 0),
                    'priority': rule.get('priority', 0),
                    'description': rule.get('description', '')
                }
                triggered_rules.append(rule_info)
                total_weight += rule.get('weight', 0)
                
                # 统计优先级
                priority = rule.get('priority', 0)
                if priority >= 90:
                    high_priority_count += 1
                elif priority >= 70:
                    medium_priority_count += 1
                
                # 收集原因
                action = rule.get('action', {})
                reason = action.get('reason', rule.get('description', ''))
                if reason and reason not in reasons:
                    reasons.append(reason)
        
        # 决策逻辑 - 基于配置的策略
        decision = self._make_decision(
            total_weight, 
            high_priority_count, 
            medium_priority_count,
            triggered_rules
        )
        
        return {
            'triggered': len(triggered_rules) > 0,
            'triggered_rules': triggered_rules,
            'risk_score': min(total_weight, 1.0),
            'decision': decision,
            'reasons': reasons
        }
    
    def _make_decision(self, total_weight: float, high_priority_count: int, 
                       medium_priority_count: int, triggered_rules: List[Dict]) -> str:
        """根据策略配置做出决策"""
        
        # 检查是否有直接拒绝的规则
        for rule in triggered_rules:
            action = next((r.get('action', {}) for r in self.active_rules 
                          if r.get('rule_id') == rule['rule_id']), {})
            if action.get('decision') == 'reject':
                return 'reject'
        
        # 基于策略配置
        auto_reject_count = self.rule_strategy.get('auto_reject_high_priority_count', 2)
        auto_review_count = self.rule_strategy.get('auto_review_medium_priority_count', 3)
        reject_threshold = self.rule_strategy.get('reject_weight_threshold', 1.5)
        review_threshold = self.rule_strategy.get('review_weight_threshold', 0.8)
        
        if high_priority_count >= auto_reject_count:
            return 'reject'
        
        if total_weight >= reject_threshold:
            return 'reject'
        
        if medium_priority_count >= auto_review_count:
            return 'review'
        
        if total_weight >= review_threshold:
            return 'review'
        
        return 'pass'
    
    def _check_rule_conditions(self, rule: Dict[str, Any], request: FraudDetectionRequest) -> bool:
        """检查规则的所有条件是否满足"""
        conditions = rule.get('conditions', {})
        if not conditions:
            return False
        
        try:
            # 金额条件
            if 'amount_gt' in conditions and request.amount <= conditions['amount_gt']:
                return False
            if 'amount_gte' in conditions and request.amount < conditions['amount_gte']:
                return False
            if 'amount_lt' in conditions and request.amount >= conditions['amount_lt']:
                return False
            if 'amount_lte' in conditions and request.amount > conditions['amount_lte']:
                return False
            
            # 频率条件 - 需要查询数据库
            if 'count_1m_gt' in conditions:
                count = self._get_transaction_count(request.user_id, minutes=1)
                if count <= conditions['count_1m_gt']:
                    return False
            
            if 'count_1h_gt' in conditions:
                count = self._get_transaction_count(request.user_id, minutes=60)
                if count <= conditions['count_1h_gt']:
                    return False
            
            # 时间条件
            if 'time_between' in conditions:
                current_time = datetime.now().time()
                time_range = conditions['time_between']
                start_time = datetime.strptime(time_range['start'], '%H:%M').time()
                end_time = datetime.strptime(time_range['end'], '%H:%M').time()
                
                if not (start_time <= current_time <= end_time):
                    return False
            
            # 设备条件
            if 'device_user_count_1h_gt' in conditions and request.device_id:
                count = self._get_device_user_count(request.device_id, minutes=60)
                if count <= conditions['device_user_count_1h_gt']:
                    return False
            
            if 'is_new_device' in conditions and conditions['is_new_device']:
                is_new = self._is_new_device(request.user_id, request.device_id)
                if not is_new:
                    return False
            
            # 黑名单条件
            if conditions.get('ip_in_blacklist') and request.ip_address:
                if not self._check_ip_blacklist(request.ip_address):
                    return False
            
            if conditions.get('device_in_blacklist') and request.device_id:
                if not self._check_device_blacklist(request.device_id):
                    return False
            
            if conditions.get('user_in_blacklist'):
                if not self._check_user_blacklist(request.user_id):
                    return False
            
            # 第一次交易条件 - 如果数据库为空，跳过此规则
            if 'is_first_transaction' in conditions and conditions['is_first_transaction']:
                count = self._get_transaction_count(request.user_id, minutes=999999)
                # 如果数据库为空或者这是第一笔，且没有历史数据，跳过
                if count == 0:
                    return False  # 没有历史数据时跳过此规则
            
            # 地理位置条件 - 需要有历史数据才能判断
            if 'location_distance_km_gt' in conditions or 'time_diff_minutes_lt' in conditions:
                # 检查是否有最近的交易记录
                count = self._get_transaction_count(request.user_id, minutes=60)
                if count == 0:
                    return False  # 没有最近交易记录，跳过地理位置规则
            
            # 失败次数条件 - 需要有失败记录
            if 'failed_count_10m_gt' in conditions:
                # TODO: 这里需要从数据库查询失败记录
                # 现在暂时跳过
                return False
            
            # 境外交易条件 - 需要 IP 地理位置数据
            if conditions.get('is_foreign'):
                # TODO: 实现 IP 地理位置查询
                # 现在暂时跳过
                return False
            
            # 所有条件都满足
            return True
            
        except Exception as e:
            logger.error(f"检查规则条件时出错: {e}")
            return False
    
    def _get_transaction_count(self, user_id: str, minutes: int) -> int:
        """获取用户在指定时间内的交易次数"""
        try:
            time_threshold = datetime.now() - timedelta(minutes=minutes)
            count = self.db.query(Transaction).filter(
                Transaction.user_id == user_id,
                Transaction.transaction_time >= time_threshold
            ).count()
            return count
        except:
            return 0
    
    def _get_device_user_count(self, device_id: str, minutes: int) -> int:
        """获取设备在指定时间内的使用用户数"""
        try:
            time_threshold = datetime.now() - timedelta(minutes=minutes)
            query = text("""
                SELECT COUNT(DISTINCT user_id) 
                FROM transactions 
                WHERE device_id = :device_id 
                AND transaction_time >= :time_threshold
            """)
            result = self.db.execute(
                query, 
                {"device_id": device_id, "time_threshold": time_threshold}
            )
            return result.scalar() or 0
        except:
            return 0
    
    def _is_new_device(self, user_id: str, device_id: str) -> bool:
        """检查是否是新设备"""
        try:
            count = self.db.query(Transaction).filter(
                Transaction.user_id == user_id,
                Transaction.device_id == device_id
            ).count()
            return count == 0
        except:
            return False
    
    def _check_ip_blacklist(self, ip_address: str) -> bool:
        """检查IP是否在黑名单中 - TODO: 实现真实的黑名单逻辑"""
        # 这里可以连接到黑名单数据库或缓存
        blacklist_ips = ['192.168.1.100', '10.0.0.1']  # 示例
        return ip_address in blacklist_ips
    
    def _check_device_blacklist(self, device_id: str) -> bool:
        """检查设备是否在黑名单中 - TODO: 实现真实的黑名单逻辑"""
        blacklist_devices = ['DEVICE_BLACKLIST_001']  # 示例
        return device_id in blacklist_devices
    
    def _check_user_blacklist(self, user_id: str) -> bool:
        """检查用户是否在黑名单中 - TODO: 实现真实的黑名单逻辑"""
        blacklist_users = ['USER_FRAUD_001']  # 示例
        return user_id in blacklist_users
    
    def _check_rule(self, request: FraudDetectionRequest, rule: FraudRule) -> bool:
        """检查单个规则是否触发"""
        try:
            rule_type = rule.rule_type
            condition = rule.condition
            
            # 金额规则
            if rule_type == "amount":
                return self._check_amount_rule(request, condition, rule.threshold)
            
            # 频率规则
            elif rule_type == "frequency":
                return self._check_frequency_rule(request, condition)
            
            # 地理位置规则
            elif rule_type == "location":
                return self._check_location_rule(request, condition)
            
            # 设备规则
            elif rule_type == "device":
                return self._check_device_rule(request, condition)
            
            return False
        
        except Exception as e:
            logger.error(f"规则检查失败 - 规则ID: {rule.rule_id}, 错误: {str(e)}")
            return False
    
    def _check_amount_rule(
        self, 
        request: FraudDetectionRequest, 
        condition: Dict[str, Any],
        threshold: float
    ) -> bool:
        """检查金额规则"""
        operator = condition.get("operator", ">")
        
        if operator == ">":
            return request.amount > threshold
        elif operator == ">=":
            return request.amount >= threshold
        elif operator == "<":
            return request.amount < threshold
        elif operator == "<=":
            return request.amount <= threshold
        
        return False
    
    def _check_frequency_rule(
        self, 
        request: FraudDetectionRequest, 
        condition: Dict[str, Any]
    ) -> bool:
        """检查频率规则 - 在指定时间窗口内的交易次数"""
        try:
            time_window = condition.get("time_window_minutes", 60)  # 默认1小时
            max_count = condition.get("max_count", 5)
            
            # 查询时间窗口内的交易次数
            time_threshold = datetime.now() - timedelta(minutes=time_window)
            
            # 按user_id统计
            if condition.get("check_by") == "user_id":
                count = self.db.query(Transaction).filter(
                    Transaction.user_id == int(request.user_id) if request.user_id.isdigit() else 0,
                    Transaction.transaction_time >= time_threshold
                ).count()
            # 按device_id统计
            elif condition.get("check_by") == "device_id" and request.device_id:
                count = self.db.query(Transaction).filter(
                    Transaction.device_id == request.device_id,
                    Transaction.transaction_time >= time_threshold
                ).count()
            # 按ip_address统计
            elif condition.get("check_by") == "ip_address" and request.ip_address:
                count = self.db.query(Transaction).filter(
                    Transaction.ip_address == request.ip_address,
                    Transaction.transaction_time >= time_threshold
                ).count()
            else:
                return False
            
            return count >= max_count
            
        except Exception as e:
            logger.warning(f"频率规则检查失败: {str(e)}")
            return False
    
    def _check_location_rule(
        self, 
        request: FraudDetectionRequest, 
        condition: Dict[str, Any]
    ) -> bool:
        """检查地理位置规则 - 检测异常地理位置"""
        try:
            # 如果没有地理位置信息，跳过检查
            if not request.location:
                return False
            
            # 黑名单地区
            if "blacklist_locations" in condition:
                blacklist = condition["blacklist_locations"]
                for blocked_location in blacklist:
                    if blocked_location.lower() in request.location.lower():
                        return True
            
            # 检查地理位置跳变（需要历史交易数据）
            if condition.get("check_geo_velocity", False):
                # 查询最近的交易地点
                time_threshold = datetime.now() - timedelta(hours=1)
                recent_transaction = self.db.query(Transaction).filter(
                    Transaction.user_id == int(request.user_id) if request.user_id.isdigit() else 0,
                    Transaction.transaction_time >= time_threshold,
                    Transaction.location.isnot(None)
                ).order_by(Transaction.transaction_time.desc()).first()
                
                if recent_transaction and recent_transaction.location:
                    # 简单检查：如果地点完全不同，可能存在风险
                    if recent_transaction.location != request.location:
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"地理位置规则检查失败: {str(e)}")
            return False
    
    def _check_device_rule(
        self, 
        request: FraudDetectionRequest, 
        condition: Dict[str, Any]
    ) -> bool:
        """检查设备规则 - 检测异常设备行为"""
        try:
            # 如果没有设备信息，跳过检查
            if not request.device_id:
                return False
            
            # 检查设备是否在黑名单
            if "blacklist_devices" in condition:
                if request.device_id in condition["blacklist_devices"]:
                    return True
            
            # 检查设备关联的用户数量（一个设备多个用户可能有风险）
            if condition.get("check_multi_user", False):
                max_users = condition.get("max_users", 3)
                
                # 查询该设备关联的不同用户数
                user_count = self.db.query(Transaction.user_id).filter(
                    Transaction.device_id == request.device_id
                ).distinct().count()
                
                if user_count >= max_users:
                    return True
            
            # 检查设备更换频率（用户频繁更换设备）
            if condition.get("check_device_switch", False):
                time_window = condition.get("time_window_days", 7)
                max_devices = condition.get("max_devices", 5)
                
                time_threshold = datetime.now() - timedelta(days=time_window)
                device_count = self.db.query(Transaction.device_id).filter(
                    Transaction.user_id == int(request.user_id) if request.user_id.isdigit() else 0,
                    Transaction.transaction_time >= time_threshold
                ).distinct().count()
                
                if device_count >= max_devices:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"设备规则检查失败: {str(e)}")
            return False


def create_default_rules(db: Session):
    """创建默认规则"""
    default_rules = [
        {
            "rule_id": "RULE_AMOUNT_HIGH",
            "rule_name": "超大额交易",
            "rule_type": "amount",
            "description": "单笔交易金额超过10万元",
            "condition": {"operator": ">"},
            "threshold": 100000.0,
            "weight": 0.5,
            "priority": 10,
        },
        {
            "rule_id": "RULE_AMOUNT_EXTREME",
            "rule_name": "极大额交易",
            "rule_type": "amount",
            "description": "单笔交易金额超过50万元",
            "condition": {"operator": ">"},
            "threshold": 500000.0,
            "weight": 0.8,
            "priority": 20,
        },
    ]
    
    for rule_data in default_rules:
        existing = db.query(FraudRule).filter(
            FraudRule.rule_id == rule_data["rule_id"]
        ).first()
        
        if not existing:
            rule = FraudRule(**rule_data, is_active=True)
            db.add(rule)
    
    db.commit()
    logger.info("默认规则创建完成")
