"""
测试规则引擎功能

通过 HTTP 请求测试不同场景下规则的触发情况
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1"


def test_rule_scenarios():
    """测试不同的规则场景"""
    
    print("\n" + "="*80)
    print("🧪 测试规则引擎功能")
    print("="*80)
    
    # 测试场景
    scenarios = [
        {
            "name": "场景1: 小额正常交易 (应该通过)",
            "data": {
                "transaction_id": "RULE_TEST_001",
                "user_id": "USER_001",
                "amount": 100.00,
                "merchant_id": "M001",
                "device_id": "DEVICE_NORMAL",
                "ip_address": "192.168.1.1",
                "location": "北京",
                "detection_mode": "fast"
            },
            "expected": "pass"
        },
        {
            "name": "场景2: 大额交易 (触发金额规则)",
            "data": {
                "transaction_id": "RULE_TEST_002",
                "user_id": "USER_002",
                "amount": 6000.00,
                "merchant_id": "M002",
                "device_id": "DEVICE_NORMAL",
                "ip_address": "192.168.1.2",
                "location": "上海",
                "detection_mode": "fast"
            },
            "expected": "review"
        },
        {
            "name": "场景3: 超大额交易 (触发超大额规则 - 应拒绝)",
            "data": {
                "transaction_id": "RULE_TEST_003",
                "user_id": "USER_003",
                "amount": 15000.00,
                "merchant_id": "M003",
                "device_id": "DEVICE_NORMAL",
                "ip_address": "192.168.1.3",
                "location": "深圳",
                "detection_mode": "fast"
            },
            "expected": "reject"
        },
        {
            "name": "场景4: IP黑名单测试",
            "data": {
                "transaction_id": "RULE_TEST_004",
                "user_id": "USER_004",
                "amount": 500.00,
                "merchant_id": "M004",
                "device_id": "DEVICE_NORMAL",
                "ip_address": "192.168.1.100",  # 黑名单IP
                "location": "杭州",
                "detection_mode": "fast"
            },
            "expected": "reject"
        },
        {
            "name": "场景5: 设备黑名单测试",
            "data": {
                "transaction_id": "RULE_TEST_005",
                "user_id": "USER_005",
                "amount": 500.00,
                "merchant_id": "M005",
                "device_id": "DEVICE_BLACKLIST_001",  # 黑名单设备
                "ip_address": "192.168.1.5",
                "location": "广州",
                "detection_mode": "fast"
            },
            "expected": "reject"
        },
    ]
    
    # 执行测试
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─'*80}")
        print(f"📋 {scenario['name']}")
        print(f"{'─'*80}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/fraud/detect",
                json=scenario['data'],
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 打印结果
                print(f"交易ID: {result['transaction_id']}")
                print(f"欺诈分数: {result['fraud_score']:.3f}")
                print(f"风险等级: {result['risk_level']}")
                print(f"是否欺诈: {'是' if result['is_fraud'] else '否'}")
                print(f"检测方法: {result['detection_method']}")
                print(f"执行时间: {result['execution_time_ms']:.2f}ms")
                print(f"建议: {result['recommendation']}")
                
                if result.get('risk_factors'):
                    print(f"风险因素:")
                    for factor in result['risk_factors']:
                        print(f"  • {factor}")
                
                # 判断测试是否通过
                expected = scenario.get('expected', 'pass')
                actual_decision = 'pass'
                if result['is_fraud']:
                    if result['fraud_score'] >= 0.8 or '拒绝' in result.get('recommendation', ''):
                        actual_decision = 'reject'
                    else:
                        actual_decision = 'review'
                elif result['risk_level'] == 'medium' or '验证' in result.get('recommendation', ''):
                    actual_decision = 'review'
                
                test_passed = (actual_decision == expected)
                
                results.append({
                    'scenario': scenario['name'],
                    'expected': expected,
                    'actual': actual_decision,
                    'passed': test_passed,
                    'fraud_score': result['fraud_score'],
                    'risk_factors': len(result.get('risk_factors', []))
                })
                
                # 打印测试结果
                if test_passed:
                    print(f"\n✅ 测试通过 (预期: {expected}, 实际: {actual_decision})")
                else:
                    print(f"\n❌ 测试失败 (预期: {expected}, 实际: {actual_decision})")
                    
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                print(response.text)
                results.append({
                    'scenario': scenario['name'],
                    'passed': False,
                    'error': f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"❌ 异常: {str(e)}")
            results.append({
                'scenario': scenario['name'],
                'passed': False,
                'error': str(e)
            })
    
    # 打印总结
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for r in results if r.get('passed', False))
    failed = total - passed
    
    print(f"\n总测试数: {total}")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    print(f"通过率: {passed/total*100:.1f}%")
    
    if failed > 0:
        print("\n失败的测试:")
        for r in results:
            if not r.get('passed', False):
                print(f"  • {r['scenario']}")
                if 'error' in r:
                    print(f"    错误: {r['error']}")
                elif 'expected' in r:
                    print(f"    预期: {r['expected']}, 实际: {r.get('actual', 'unknown')}")


def test_yaml_config():
    """测试YAML规则配置是否正确加载"""
    print("\n" + "="*80)
    print("📄 检查 YAML 规则配置")
    print("="*80)
    
    import yaml
    from pathlib import Path
    
    config_path = Path("backend/config/fraud_rules.yaml")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        rules = config.get('rules', [])
        strategy = config.get('rule_strategy', {})
        
        print(f"\n✅ 成功加载配置文件")
        print(f"规则总数: {len(rules)}")
        print(f"激活规则: {sum(1 for r in rules if r.get('is_active', True))}")
        
        # 按类型统计
        rule_types = {}
        for rule in rules:
            rule_type = rule.get('rule_type', 'unknown')
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
        
        print(f"\n规则类型分布:")
        for rule_type, count in sorted(rule_types.items()):
            print(f"  • {rule_type}: {count}条")
        
        print(f"\n规则策略配置:")
        print(f"  • 高优先级自动拒绝阈值: {strategy.get('auto_reject_high_priority_count')}")
        print(f"  • 中优先级自动审核阈值: {strategy.get('auto_review_medium_priority_count')}")
        print(f"  • 拒绝权重阈值: {strategy.get('reject_weight_threshold')}")
        print(f"  • 审核权重阈值: {strategy.get('review_weight_threshold')}")
        
    except Exception as e:
        print(f"❌ 加载配置失败: {str(e)}")


if __name__ == "__main__":
    # 先检查配置文件
    test_yaml_config()
    
    # 再测试规则功能
    test_rule_scenarios()
    
    print("\n" + "="*80)
    print("✨ 测试完成!")
    print("="*80)
