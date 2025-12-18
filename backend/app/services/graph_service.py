"""
图特征提取服务

提供基于图数据库(模拟)的特征提取功能
实际应用中应连接Neo4j,这里使用模拟数据
"""
import numpy as np
from typing import Dict, Any, Optional
from app.core.logger import logger


class GraphFeatureService:
    """图特征提取服务"""
    
    def __init__(self):
        """初始化服务"""
        self.enabled = True
        logger.info("图特征服务已初始化 (模拟模式)")
    
    def extract_graph_features(
        self,
        user_id: str,
        device_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        amount: float = 0.0
    ) -> np.ndarray:
        """
        提取15维图特征
        
        在实际生产环境中,这里应该:
        1. 查询 Redis 缓存
        2. 如果缓存未命中,查询 Neo4j
        3. 将结果写入 Redis 缓存
        
        当前实现使用模拟数据+规则逻辑
        
        Returns:
            15维图特征向量
        """
        try:
            # 模拟图特征计算
            graph_features = []
            
            # ===== 节点特征 (5维) =====
            
            # 1. 关联欺诈节点数量 (基于user_id hash)
            user_hash = hash(user_id) % 10
            fraud_neighbor_count = user_hash if user_hash < 3 else 0
            graph_features.append(fraud_neighbor_count)
            
            # 2. 用户社区欺诈率 (基于模式)
            community_fraud_rate = 0.15 if user_hash > 5 else 0.05
            graph_features.append(community_fraud_rate)
            
            # 3. 用户度中心性 (连接数)
            degree_centrality = min(user_hash * 2, 20)
            graph_features.append(degree_centrality)
            
            # 4. PageRank分数 (基于连接模式)
            pagerank = 0.1 + (user_hash / 100)
            graph_features.append(pagerank)
            
            # 5. 到最近欺诈节点距离
            distance_to_fraud = 3 if fraud_neighbor_count == 0 else 1
            graph_features.append(distance_to_fraud)
            
            # ===== 关系特征 (7维) =====
            
            # 6-7. 设备共享度
            if device_id:
                device_hash = hash(device_id) % 10
                device_share_count = min(device_hash, 10)
                device_fraud_rate = 0.3 if device_hash > 7 else 0.1
            else:
                device_share_count = 1
                device_fraud_rate = 0.0
            graph_features.append(device_share_count)
            graph_features.append(device_fraud_rate)
            
            # 8-9. IP共享度
            if ip_address:
                ip_hash = hash(ip_address) % 10
                ip_share_count = min(ip_hash, 8)
                ip_fraud_rate = 0.2 if ip_hash > 6 else 0.05
            else:
                ip_share_count = 1
                ip_fraud_rate = 0.0
            graph_features.append(ip_share_count)
            graph_features.append(ip_fraud_rate)
            
            # 10. 地址共享度 (简化,使用user_id)
            address_share_count = min(user_hash, 5)
            graph_features.append(address_share_count)
            
            # 11. 最强关联权重
            max_relationship_strength = 0.8 if fraud_neighbor_count > 0 else 0.3
            graph_features.append(max_relationship_strength)
            
            # 12. 欺诈聚集系数
            fraud_cluster_coef = community_fraud_rate
            graph_features.append(fraud_cluster_coef)
            
            # ===== 子图特征 (3维) =====
            
            # 13. 局部聚集系数
            local_clustering = 0.6 if degree_centrality > 10 else 0.3
            graph_features.append(local_clustering)
            
            # 14. 1跳邻居欺诈率
            neighbor_fraud_rate = community_fraud_rate * 1.2
            graph_features.append(neighbor_fraud_rate)
            
            # 15. 是否在欺诈社区 (0/1)
            is_fraud_community = 1.0 if community_fraud_rate > 0.1 else 0.0
            graph_features.append(is_fraud_community)
            
            # 转换为numpy数组
            features_array = np.array(graph_features, dtype=np.float32)
            
            logger.debug(f"提取图特征成功 - user_id: {user_id}, 特征维度: {len(features_array)}")
            
            return features_array
            
        except Exception as e:
            logger.error(f"图特征提取失败: {str(e)}")
            # 返回零向量
            return np.zeros(15, dtype=np.float32)
    
    def get_feature_names(self) -> list:
        """获取特征名称列表"""
        return [
            # 节点特征
            'fraud_neighbor_count',
            'community_fraud_rate',
            'degree_centrality',
            'pagerank_score',
            'distance_to_fraud',
            # 关系特征
            'device_share_count',
            'device_fraud_rate',
            'ip_share_count',
            'ip_fraud_rate',
            'address_share_count',
            'max_relationship_strength',
            'fraud_cluster_coefficient',
            # 子图特征
            'local_clustering_coefficient',
            'neighbor_fraud_rate',
            'is_in_fraud_community'
        ]


# 创建全局实例
graph_feature_service = GraphFeatureService()
