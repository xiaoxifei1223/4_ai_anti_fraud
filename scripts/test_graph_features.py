"""
测试图特征融合功能
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'backend'))

import numpy as np
from app.services.feature_service import extract_features, get_feature_names
from app.services.graph_service import GraphFeatureService
from app.models.schemas import FraudDetectionRequest
from app.db.database import SessionLocal, init_db

def test_feature_names():
    """测试特征名称列表"""
    print("\n=== 测试特征名称 ===")
    
    feature_names = get_feature_names()
    print(f"特征总数: {len(feature_names)}")
    print(f"预期: 45维 (表格30维 + 图15维)")
    
    print("\n表格特征 (前30维):")
    for i, name in enumerate(feature_names[:30]):
        print(f"  {i+1}. {name}")
    
    print("\n图特征 (后15维):")
    for i, name in enumerate(feature_names[30:]):
        print(f"  {31+i}. {name}")
    
    assert len(feature_names) == 45, f"特征数量错误,期望45,实际{len(feature_names)}"
    print("\n✓ 特征名称测试通过!")


def test_graph_service():
    """测试图特征服务"""
    print("\n=== 测试图特征服务 ===")
    
    graph_service = GraphFeatureService()
    
    # 测试提取图特征
    graph_features = graph_service.extract_graph_features(
        user_id="test_user_001",
        device_id="device_12345",
        ip_address="192.168.1.100",
        amount=5000.0
    )
    
    print(f"图特征维度: {len(graph_features)}")
    print(f"图特征类型: {type(graph_features)}")
    print(f"图特征数据类型: {graph_features.dtype}")
    
    # 显示图特征值
    feature_names = graph_service.get_feature_names()
    print("\n图特征值:")
    for name, value in zip(feature_names, graph_features):
        print(f"  {name}: {value:.4f}")
    
    assert len(graph_features) == 15, f"图特征维度错误,期望15,实际{len(graph_features)}"
    assert graph_features.dtype == np.float32, f"数据类型错误,期望float32,实际{graph_features.dtype}"
    print("\n✓ 图特征服务测试通过!")


def test_feature_extraction():
    """测试完整特征提取"""
    print("\n=== 测试完整特征提取 ===")
    
    # 初始化数据库
    init_db()
    db = SessionLocal()
    
    try:
        # 创建测试请求
        request = FraudDetectionRequest(
            transaction_id="test_txn_001",
            user_id="test_user_001",
            amount=5000.0,
            merchant_id="merchant_123",
            device_id="device_12345",
            ip_address="192.168.1.100"
        )
        
        # 测试启用图特征
        print("\n测试1: 启用图特征")
        features_with_graph = extract_features(request, db, use_graph=True)
        print(f"  特征维度: {len(features_with_graph)}")
        print(f"  特征类型: {type(features_with_graph)}")
        print(f"  数据类型: {features_with_graph.dtype}")
        print(f"  特征范围: [{features_with_graph.min():.4f}, {features_with_graph.max():.4f}]")
        
        assert len(features_with_graph) == 45, f"特征维度错误,期望45,实际{len(features_with_graph)}"
        
        # 测试禁用图特征
        print("\n测试2: 禁用图特征")
        features_without_graph = extract_features(request, db, use_graph=False)
        print(f"  特征维度: {len(features_without_graph)}")
        print(f"  后15维应为零向量: {features_without_graph[-15:]}")
        
        assert len(features_without_graph) == 45, f"特征维度错误,期望45,实际{len(features_without_graph)}"
        assert np.allclose(features_without_graph[-15:], 0.0), "禁用图特征时后15维应为零向量"
        
        # 对比前30维应相同
        print("\n测试3: 对比表格特征")
        table_features_1 = features_with_graph[:30]
        table_features_2 = features_without_graph[:30]
        
        print(f"  启用图特征的表格特征: {table_features_1[:5]}...")
        print(f"  禁用图特征的表格特征: {table_features_2[:5]}...")
        
        assert np.allclose(table_features_1, table_features_2), "表格特征应保持一致"
        
        print("\n✓ 完整特征提取测试通过!")
        
    finally:
        db.close()


def test_feature_consistency():
    """测试不同用户的特征一致性"""
    print("\n=== 测试特征一致性 ===")
    
    graph_service = GraphFeatureService()
    
    # 测试相同用户多次请求
    print("\n测试1: 相同用户的特征一致性")
    features_1 = graph_service.extract_graph_features(
        user_id="test_user_001",
        device_id="device_12345",
        ip_address="192.168.1.100",
        amount=5000.0
    )
    
    features_2 = graph_service.extract_graph_features(
        user_id="test_user_001",
        device_id="device_12345",
        ip_address="192.168.1.100",
        amount=5000.0
    )
    
    assert np.allclose(features_1, features_2), "相同输入应产生相同特征"
    print("  ✓ 相同用户特征一致")
    
    # 测试不同用户
    print("\n测试2: 不同用户的特征差异")
    features_3 = graph_service.extract_graph_features(
        user_id="test_user_002",
        device_id="device_67890",
        ip_address="192.168.1.200",
        amount=10000.0
    )
    
    # 不同用户的某些特征应该不同
    diff_count = np.sum(~np.isclose(features_1, features_3))
    print(f"  差异特征数量: {diff_count}/15")
    assert diff_count > 0, "不同用户应有不同特征"
    print("  ✓ 不同用户特征有差异")
    
    print("\n✓ 特征一致性测试通过!")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("图特征融合功能测试")
    print("=" * 60)
    
    try:
        # 测试1: 特征名称
        test_feature_names()
        
        # 测试2: 图特征服务
        test_graph_service()
        
        # 测试3: 完整特征提取
        test_feature_extraction()
        
        # 测试4: 特征一致性
        test_feature_consistency()
        
        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        
        # 显示总结
        print("\n总结:")
        print(f"  ✓ 特征总维度: 45维")
        print(f"  ✓ 表格特征: 30维")
        print(f"  ✓ 图特征: 15维")
        print(f"  ✓ 特征融合: 成功")
        print(f"  ✓ 特征一致性: 通过")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"测试失败: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
