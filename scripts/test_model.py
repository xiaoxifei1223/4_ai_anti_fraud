"""
测试 XGBoost 模型训练结果

通过 HTTP 请求测试实际欺诈检测功能
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def test_health():
    """测试健康检查"""
    print("\n" + "="*60)
    print("🔍 测试健康检查接口")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    
    print(f"状态: {data['status']}")
    print(f"版本: {data['version']}")
    print(f"数据库: {data['database']}")
    print(f"模型已加载: {data['model_loaded']}")
    
    return data['model_loaded']


def test_fraud_detection():
    """测试欺诈检测"""
    print("\n" + "="*60)
    print("🎯 测试欺诈检测接口")
    print("="*60)
    
    # 测试用例
    test_cases = [
        {
            "name": "小额正常交易",
            "data": {
                "transaction_id": "T001",
                "user_id": "U123",
                "amount": 100.0,
                "transaction_type": "purchase",
                "merchant_id": "M001",
                "device_id": "D001",
                "ip_address": "192.168.1.1",
                "detection_mode": "fast"
            }
        },
        {
            "name": "大额可疑交易",
            "data": {
                "transaction_id": "T002",
                "user_id": "U456",
                "amount": 80000.0,
                "transaction_type": "transfer",
                "merchant_id": "M002",
                "device_id": "D002",
                "ip_address": "10.0.0.1",
                "detection_mode": "fast"
            }
        },
        {
            "name": "超大额交易",
            "data": {
                "transaction_id": "T003",
                "user_id": "U789",
                "amount": 200000.0,
                "transaction_type": "withdrawal",
                "merchant_id": "M003",
                "device_id": "D003",
                "ip_address": "172.16.0.1",
                "detection_mode": "fast"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n【{test_case['name']}】")
        print(f"  交易ID: {test_case['data']['transaction_id']}")
        print(f"  金额: ¥{test_case['data']['amount']:,.2f}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/fraud/detect",
                json=test_case['data']
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ 检测成功")
                print(f"  欺诈分数: {result['fraud_score']:.4f}")
                print(f"  风险等级: {result['risk_level']}")
                print(f"  是否欺诈: {'是' if result['is_fraud'] else '否'}")
                print(f"  风险因素: {', '.join(result['risk_factors']) if result['risk_factors'] else '无'}")
                print(f"  执行时间: {result['execution_time_ms']:.2f}ms")
                print(f"  建议: {result['recommendation']}")
            else:
                print(f"  ❌ 检测失败: {response.status_code}")
                print(f"  错误信息: {response.text}")
        
        except Exception as e:
            print(f"  ❌ 请求失败: {str(e)}")


def main():
    """主函数"""
    print("\n🚀 XGBoost 模型测试开始")
    print("="*60)
    
    # 1. 检查服务健康状态
    model_loaded = test_health()
    
    if not model_loaded:
        print("\n⚠️  模型未加载，无法测试")
        return
    
    # 2. 测试欺诈检测
    test_fraud_detection()
    
    print("\n" + "="*60)
    print("✅ 测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
