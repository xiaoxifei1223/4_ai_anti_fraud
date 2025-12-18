"""
加载图数据集 (YelpChi/Amazon)

从预处理的.npz文件中加载图数据,用于模拟图特征提取
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class GraphDataLoader:
    """图数据加载器"""
    
    def __init__(self, dataset_name='yelp'):
        """
        初始化图数据加载器
        
        Args:
            dataset_name: 'yelp' 或 'amazon'
        """
        self.dataset_name = dataset_name
        # 数据在 data/processed/graph/ 目录下
        self.data_dir = project_root / 'data' / 'processed' / 'graph' / dataset_name
        self.graph_data = None
        
    def load(self):
        """加载图数据"""
        print(f"\n{'='*60}")
        print(f"加载 {self.dataset_name.upper()} 图数据集")
        print(f"{'='*60}")
        
        # 尝试加载.npz文件，如果不存在则从分离文件加载
        data_file = self.data_dir / f'{self.dataset_name}.npz'
        
        if data_file.exists():
            # 直接加载.npz文件
            print(f"\n📂 加载文件: {data_file}")
            data = np.load(str(data_file), allow_pickle=True)
            
            self.graph_data = {
                'features': data['features'],
                'labels': data['labels'],
                'adjacency': data['adjacency'],
            }
        else:
            # 从分离文件加载
            print(f"\n📂 从分离文件加载...")
            features_file = self.data_dir / 'features.npy'
            labels_file = self.data_dir / 'labels.npy'
            adj_file = self.data_dir / 'adj_matrix.npz'
            
            if not all([f.exists() for f in [features_file, labels_file, adj_file]]):
                raise FileNotFoundError(f"数据文件不存在: {self.data_dir}")
            
            print(f"   加载: {features_file.name}")
            features = np.load(str(features_file), allow_pickle=True)
            
            print(f"   加载: {labels_file.name}")
            labels = np.load(str(labels_file), allow_pickle=True)
            
            print(f"   加载: {adj_file.name}")
            adj_data = np.load(str(adj_file), allow_pickle=True)
            adjacency = csr_matrix(
                (adj_data['data'], adj_data['indices'], adj_data['indptr']),
                shape=adj_data['shape']
            )
            
            self.graph_data = {
                'features': features,
                'labels': labels,
                'adjacency': adjacency,
            }
        
        # 转换邻接矩阵为稀疏矩阵
        if not isinstance(self.graph_data['adjacency'], csr_matrix):
            self.graph_data['adjacency'] = csr_matrix(self.graph_data['adjacency'])
        
        # 打印统计信息
        n_nodes = self.graph_data['features'].shape[0]
        n_features = self.graph_data['features'].shape[1]
        n_edges = self.graph_data['adjacency'].nnz
        n_fraud = np.sum(self.graph_data['labels'] == 1)
        fraud_rate = n_fraud / n_nodes * 100
        
        print(f"\n📊 数据统计:")
        print(f"  节点数量: {n_nodes:,}")
        print(f"  节点特征维度: {n_features}")
        print(f"  边数量: {n_edges:,}")
        print(f"  平均度: {n_edges / n_nodes:.2f}")
        print(f"  欺诈节点数: {n_fraud:,}")
        print(f"  欺诈率: {fraud_rate:.2f}%")
        
        return self.graph_data
    
    def get_node_info(self, node_id):
        """获取节点信息"""
        if self.graph_data is None:
            raise ValueError("请先调用 load() 方法加载数据")
        
        if node_id >= self.graph_data['features'].shape[0]:
            return None
        
        # 获取邻居节点
        adjacency = self.graph_data['adjacency']
        neighbors = adjacency[node_id].nonzero()[1]
        
        # 统计邻居中的欺诈节点
        neighbor_labels = self.graph_data['labels'][neighbors]
        n_fraud_neighbors = np.sum(neighbor_labels == 1)
        
        return {
            'node_id': node_id,
            'is_fraud': bool(self.graph_data['labels'][node_id] == 1),
            'features': self.graph_data['features'][node_id],
            'degree': len(neighbors),
            'fraud_neighbors': n_fraud_neighbors,
            'neighbor_fraud_rate': n_fraud_neighbors / len(neighbors) if len(neighbors) > 0 else 0.0
        }
    
    def compute_graph_features(self, node_id):
        """
        计算节点的图特征 (模拟15维图特征)
        
        这里使用简化的图特征计算,实际应该连接Neo4j
        """
        node_info = self.get_node_info(node_id)
        
        if node_info is None:
            # 节点不存在,返回零特征
            return np.zeros(15)
        
        adjacency = self.graph_data['adjacency']
        neighbors = adjacency[node_id].nonzero()[1]
        
        # 计算图特征
        features = []
        
        # 1. 节点特征 (5维)
        features.append(node_info['fraud_neighbors'])  # 欺诈邻居数量
        features.append(node_info['neighbor_fraud_rate'])  # 邻居欺诈率
        features.append(node_info['degree'])  # 节点度
        features.append(0.0)  # PageRank (简化,使用0)
        
        # 计算到欺诈节点的最短距离 (限制2跳)
        fraud_nodes = np.where(self.graph_data['labels'] == 1)[0]
        min_distance = 999
        if len(neighbors) > 0:
            # 检查1跳邻居
            if node_info['fraud_neighbors'] > 0:
                min_distance = 1
            else:
                # 检查2跳邻居
                for neighbor in neighbors[:10]:  # 限制检查数量
                    neighbor_neighbors = adjacency[neighbor].nonzero()[1]
                    if np.any(np.isin(neighbor_neighbors, fraud_nodes)):
                        min_distance = 2
                        break
        features.append(min_distance)
        
        # 2. 关系特征 (7维) - 简化计算
        # 这里使用度数来模拟共享度
        features.append(min(node_info['degree'], 10))  # 设备共享度 (capped at 10)
        features.append(node_info['neighbor_fraud_rate'])  # 设备欺诈占比
        features.append(min(node_info['degree'], 8))  # IP共享度
        features.append(node_info['neighbor_fraud_rate'] * 0.8)  # IP欺诈占比
        features.append(min(node_info['degree'], 5))  # 地址共享度
        features.append(0.5)  # 最强关联权重
        features.append(node_info['neighbor_fraud_rate'])  # 欺诈聚集系数
        
        # 3. 子图特征 (3维)
        # 计算局部聚集系数
        if len(neighbors) > 1:
            neighbor_connections = 0
            for i, n1 in enumerate(neighbors[:10]):
                for n2 in neighbors[i+1:10]:
                    if adjacency[n1, n2] > 0:
                        neighbor_connections += 1
            max_connections = len(neighbors) * (len(neighbors) - 1) / 2
            clustering = neighbor_connections / max_connections if max_connections > 0 else 0
        else:
            clustering = 0
        
        features.append(clustering)  # 局部聚集系数
        features.append(node_info['neighbor_fraud_rate'])  # 1跳邻居欺诈率
        features.append(1.0 if node_info['neighbor_fraud_rate'] > 0.5 else 0.0)  # 是否在欺诈社区
        
        return np.array(features)
    
    def create_graph_feature_cache(self, output_file='graph_features_cache.npz'):
        """
        为所有节点预计算图特征并缓存
        
        这模拟了离线预计算+Redis缓存的场景
        """
        print(f"\n{'='*60}")
        print("预计算图特征缓存")
        print(f"{'='*60}")
        
        n_nodes = self.graph_data['features'].shape[0]
        graph_features_list = []
        
        print(f"\n计算 {n_nodes:,} 个节点的图特征...")
        
        for i in range(n_nodes):
            if (i + 1) % 5000 == 0:
                print(f"  进度: {i+1:,}/{n_nodes:,} ({(i+1)/n_nodes*100:.1f}%)")
            
            graph_features = self.compute_graph_features(i)
            graph_features_list.append(graph_features)
        
        # 转换为数组
        graph_features_array = np.array(graph_features_list)
        
        # 保存到文件
        cache_file = self.data_dir / output_file
        np.savez_compressed(
            cache_file,
            graph_features=graph_features_array,
            node_ids=np.arange(n_nodes)
        )
        
        print(f"\n✅ 图特征缓存已保存: {cache_file}")
        print(f"   形状: {graph_features_array.shape}")
        print(f"   大小: {cache_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return graph_features_array


def main():
    """主函数"""
    # 加载 YelpChi 数据集
    print("\n" + "="*60)
    print("图数据加载与特征预计算")
    print("="*60)
    
    # 加载 Yelp 数据
    yelp_loader = GraphDataLoader('yelp')
    yelp_data = yelp_loader.load()
    
    # 测试单个节点
    print(f"\n{'='*60}")
    print("测试节点查询")
    print(f"{'='*60}")
    
    test_node_id = 100
    node_info = yelp_loader.get_node_info(test_node_id)
    print(f"\n节点 {test_node_id} 信息:")
    print(f"  是否欺诈: {node_info['is_fraud']}")
    print(f"  节点度: {node_info['degree']}")
    print(f"  欺诈邻居数: {node_info['fraud_neighbors']}")
    print(f"  邻居欺诈率: {node_info['neighbor_fraud_rate']:.2%}")
    
    # 计算图特征
    graph_features = yelp_loader.compute_graph_features(test_node_id)
    print(f"\n图特征向量 ({len(graph_features)}维):")
    print(f"  {graph_features}")
    
    # 预计算并缓存所有节点的图特征
    print(f"\n{'='*60}")
    choice = input("是否预计算所有节点的图特征? (y/n): ")
    if choice.lower() == 'y':
        yelp_loader.create_graph_feature_cache()
    
    print(f"\n{'='*60}")
    print("✨ 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
