"""
数据预处理脚本
处理 Credit Card、YelpChi 和 Amazon 数据集
"""

import polars as pl
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import pickle
import json

# 设置路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GRAPH_DIR = DATA_DIR / "graph"
PROCESSED_DIR = DATA_DIR / "processed"

# 创建输出目录
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "creditcard").mkdir(exist_ok=True)
(PROCESSED_DIR / "graph").mkdir(exist_ok=True)


def process_creditcard_data():
    """
    处理 Kaggle Credit Card Fraud Detection 数据集
    """
    print("\n" + "="*60)
    print("📊 处理 Credit Card 数据集")
    print("="*60)
    
    # 1. 读取数据
    print("\n1️⃣ 读取数据...")
    csv_path = RAW_DIR / "creditcard" / "creditcard.csv"
    
    # 使用 Polars 读取，明确指定 Time 和 Amount 为浮点数
    df = pl.read_csv(
        str(csv_path),
        schema_overrides={
            "Time": pl.Float64,
            "Amount": pl.Float64
        }
    )
    
    print(f"   ✅ 数据形状: {df.shape}")
    print(f"   📊 总交易数: {len(df):,}")
    print(f"   ⚠️  欺诈交易: {df.filter(pl.col('Class') == 1).shape[0]:,}")
    print(f"   ✅ 正常交易: {df.filter(pl.col('Class') == 0).shape[0]:,}")
    
    fraud_rate = df.filter(pl.col('Class') == 1).shape[0] / len(df) * 100
    print(f"   📈 欺诈率: {fraud_rate:.4f}%")
    
    # 2. 基础统计
    print("\n2️⃣ 数据统计...")
    print(df.describe())
    
    # 3. 特征标准化（Amount 需要标准化，V1-V28 已经 PCA 处理过）
    print("\n3️⃣ 特征标准化...")
    
    # 转换为 NumPy 进行标准化
    df_np = df.to_numpy()
    
    # Amount 在倒数第二列
    scaler = StandardScaler()
    df_np[:, -2] = scaler.fit_transform(df_np[:, -2].reshape(-1, 1)).flatten()
    
    # Time 特征转换（转为小时）
    df_np[:, 0] = df_np[:, 0] / 3600  # 秒转小时
    
    # 保存 scaler
    scaler_path = PROCESSED_DIR / "creditcard" / "amount_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler 已保存: {scaler_path.name}")
    
    # 4. 分割数据集
    print("\n4️⃣ 分割数据集...")
    
    X = df_np[:, :-1]  # 所有特征
    y = df_np[:, -1]   # 标签
    
    # 分层抽样（保持欺诈率一致）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   📊 训练集: {len(X_train):,} (欺诈: {int(y_train.sum()):,})")
    print(f"   📊 验证集: {len(X_val):,} (欺诈: {int(y_val.sum()):,})")
    print(f"   📊 测试集: {len(X_test):,} (欺诈: {int(y_test.sum()):,})")
    
    # 5. 保存处理后的数据
    print("\n5️⃣ 保存数据...")
    
    # 保存为 NumPy 格式（训练用）
    np.savez_compressed(
        PROCESSED_DIR / "creditcard" / "train.npz",
        X=X_train, y=y_train
    )
    np.savez_compressed(
        PROCESSED_DIR / "creditcard" / "val.npz",
        X=X_val, y=y_val
    )
    np.savez_compressed(
        PROCESSED_DIR / "creditcard" / "test.npz",
        X=X_test, y=y_test
    )
    
    # 保存特征名称
    feature_names = df.columns[:-1]  # 除了 Class
    with open(PROCESSED_DIR / "creditcard" / "feature_names.json", 'w') as f:
        json.dump(list(feature_names), f, indent=2)
    
    print(f"   ✅ 训练集已保存: train.npz")
    print(f"   ✅ 验证集已保存: val.npz")
    print(f"   ✅ 测试集已保存: test.npz")
    
    # 6. 生成数据摘要
    summary = {
        "dataset": "Credit Card Fraud Detection",
        "total_samples": int(len(df)),
        "fraud_samples": int(df.filter(pl.col('Class') == 1).shape[0]),
        "normal_samples": int(df.filter(pl.col('Class') == 0).shape[0]),
        "fraud_rate": float(fraud_rate),
        "features": len(feature_names),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "feature_names": list(feature_names)
    }
    
    with open(PROCESSED_DIR / "creditcard" / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✅ 数据摘要已保存: summary.json")
    
    print("\n✅ Credit Card 数据处理完成!")
    return summary


def process_graph_data():
    """
    处理图数据集 (YelpChi 和 Amazon)
    """
    print("\n" + "="*60)
    print("📊 处理图数据集")
    print("="*60)
    
    datasets = {
        "yelp": GRAPH_DIR / "CARE-GNN" / "data" / "yelp" / "YelpChi.mat",
        "amazon": GRAPH_DIR / "CARE-GNN" / "data" / "amazon" / "Amazon.mat"
    }
    
    summaries = {}
    
    for name, mat_path in datasets.items():
        print(f"\n{'='*60}")
        print(f"📊 处理 {name.upper()} 数据集")
        print(f"{'='*60}")
        
        # 1. 加载 .mat 文件
        print(f"\n1️⃣ 加载数据: {mat_path.name}")
        mat_data = sio.loadmat(str(mat_path))
        
        # 2. 提取数据
        print("\n2️⃣ 提取图结构...")
        
        # 不同数据集的键名可能不同
        if name == "yelp":
            features = mat_data.get('features', None)
            labels = mat_data.get('label', None)
            homo_adj = mat_data.get('homo', None)
            
        elif name == "amazon":
            features = mat_data.get('features', None)
            labels = mat_data.get('label', None)
            homo_adj = mat_data.get('homo', None)
        
        # 3. 数据统计
        print("\n3️⃣ 数据统计...")
        if features is not None:
            print(f"   📊 节点数量: {features.shape[0]:,}")
            print(f"   📊 特征维度: {features.shape[1]:,}")
        
        if labels is not None:
            labels_flat = labels.flatten()
            fraud_count = int((labels_flat == 1).sum())
            normal_count = int((labels_flat == 0).sum())
            fraud_rate = fraud_count / len(labels_flat) * 100
            
            print(f"   ⚠️  欺诈节点: {fraud_count:,}")
            print(f"   ✅ 正常节点: {normal_count:,}")
            print(f"   📈 欺诈率: {fraud_rate:.2f}%")
        
        if homo_adj is not None:
            # 计算边数（稀疏矩阵）
            if hasattr(homo_adj, 'nnz'):
                edge_count = homo_adj.nnz
            else:
                edge_count = np.count_nonzero(homo_adj)
            print(f"   🔗 边数量: {edge_count:,}")
        
        # 4. 保存处理后的数据
        print("\n4️⃣ 保存数据...")
        
        output_dir = PROCESSED_DIR / "graph" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为 NumPy 格式
        if features is not None:
            np.save(output_dir / "features.npy", features)
            print(f"   ✅ 特征已保存: features.npy")
        
        if labels is not None:
            np.save(output_dir / "labels.npy", labels)
            print(f"   ✅ 标签已保存: labels.npy")
        
        if homo_adj is not None:
            # 保存稀疏矩阵
            from scipy.sparse import save_npz
            if hasattr(homo_adj, 'tocsr'):
                save_npz(output_dir / "adj_matrix.npz", homo_adj.tocsr())
            else:
                save_npz(output_dir / "adj_matrix.npz", homo_adj)
            print(f"   ✅ 邻接矩阵已保存: adj_matrix.npz")
        
        # 5. 生成摘要
        summary = {
            "dataset": name,
            "num_nodes": int(features.shape[0]) if features is not None else 0,
            "num_features": int(features.shape[1]) if features is not None else 0,
            "num_edges": int(edge_count) if homo_adj is not None else 0,
            "fraud_count": int(fraud_count) if labels is not None else 0,
            "normal_count": int(normal_count) if labels is not None else 0,
            "fraud_rate": float(fraud_rate) if labels is not None else 0.0
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ 摘要已保存: summary.json")
        
        summaries[name] = summary
        
        print(f"\n✅ {name.upper()} 数据处理完成!")
    
    return summaries


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 反欺诈系统 - 数据预处理")
    print("="*60)
    
    # 处理 Credit Card 数据
    cc_summary = process_creditcard_data()
    
    # 处理图数据
    graph_summaries = process_graph_data()
    
    # 总结
    print("\n" + "="*60)
    print("📊 预处理总结")
    print("="*60)
    
    print("\n✅ Credit Card 数据集:")
    print(f"   - 总样本: {cc_summary['total_samples']:,}")
    print(f"   - 训练集: {cc_summary['train_size']:,}")
    print(f"   - 验证集: {cc_summary['val_size']:,}")
    print(f"   - 测试集: {cc_summary['test_size']:,}")
    print(f"   - 欺诈率: {cc_summary['fraud_rate']:.4f}%")
    
    print("\n✅ 图数据集:")
    for name, summary in graph_summaries.items():
        print(f"\n   {name.upper()}:")
        print(f"   - 节点数: {summary['num_nodes']:,}")
        print(f"   - 边数: {summary['num_edges']:,}")
        print(f"   - 欺诈率: {summary['fraud_rate']:.2f}%")
    
    # 保存总体摘要
    all_summary = {
        "creditcard": cc_summary,
        "graph": graph_summaries
    }
    
    with open(PROCESSED_DIR / "processing_summary.json", 'w') as f:
        json.dump(all_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("🎉 所有数据预处理完成!")
    print("="*60)
    print(f"\n📁 处理后的数据位置: {PROCESSED_DIR}")
    print("\n下一步:")
    print("  1. 数据探索分析: python scripts/analyze_data.py")
    print("  2. 训练基础模型: python scripts/train_model.py")
    print("  3. 构建图数据库: python scripts/build_graph_db.py")


if __name__ == "__main__":
    main()
