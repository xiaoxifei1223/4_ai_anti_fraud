"""
XGBoost 模型训练脚本

使用信用卡欺诈检测数据集训练 XGBoost 分类器
"""
import sys
import os
from pathlib import Path
import time
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    average_precision_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'backend'))
sys.path.insert(0, str(project_root))


class XGBoostTrainer:
    """XGBoost 模型训练器"""
    
    def __init__(self, data_dir: Path, model_dir: Path):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self):
        """加载预处理后的数据"""
        print("📂 加载训练数据...")
        
        # 加载训练集
        train_file = self.data_dir / "creditcard" / "train.npz"
        val_file = self.data_dir / "creditcard" / "val.npz"
        test_file = self.data_dir / "creditcard" / "test.npz"
        
        train_data = np.load(train_file)
        val_data = np.load(val_file)
        test_data = np.load(test_file)
        
        self.X_train = train_data['X']
        self.y_train = train_data['y']
        self.X_val = val_data['X']
        self.y_val = val_data['y']
        self.X_test = test_data['X']
        self.y_test = test_data['y']
        
        print(f"✅ 训练集: {self.X_train.shape}, 正例: {self.y_train.sum()}")
        print(f"✅ 验证集: {self.X_val.shape}, 正例: {self.y_val.sum()}")
        print(f"✅ 测试集: {self.X_test.shape}, 正例: {self.y_test.sum()}")
        
        # 计算正负样本比例
        fraud_ratio = self.y_train.sum() / len(self.y_train)
        print(f"📊 欺诈样本比例: {fraud_ratio*100:.3f}%")
        
        return fraud_ratio
    
    def load_synthetic_data(self, n_samples: int = 50000):
        """使用特征工程服务生成合成训练数据 (45维特征)"""
        print("📂 生成合成训练数据 (45维特征)...")
        
        from app.db.database import SessionLocal, init_db
        from app.models.schemas import FraudDetectionRequest
        from app.services.feature_service import extract_features
        
        init_db()
        db = SessionLocal()
        
        X_list = []
        y_list = []
        rng = np.random.default_rng(42)
        
        for i in range(n_samples):
            # 随机生成交易基本信息
            amount = float(np.clip(rng.lognormal(mean=8.0, sigma=1.0), 1.0, 200000.0))
            user_id = f"user_{rng.integers(0, 5000)}"
            device_id = f"device_{rng.integers(0, 2000)}"
            ip_address = f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
            merchant_id = f"merchant_{rng.integers(0, 500)}"
            merchant_category = f"MCC_{rng.integers(1000, 1100)}"
            transaction_type = "payment"
            location = "CN"
            
            request = FraudDetectionRequest(
                transaction_id=f"txn_{i}",
                user_id=user_id,
                amount=amount,
                merchant_id=merchant_id,
                merchant_category=merchant_category,
                device_id=device_id,
                ip_address=ip_address,
                location=location,
                transaction_type=transaction_type
            )
            
            # 使用线上特征工程提取45维特征
            features = extract_features(request, db, use_graph=True).astype(np.float32)
            
            if features.shape[0] != 45:
                raise ValueError(f"特征维度错误, 期望45, 实际{features.shape[0]}")
            
            # 构造合成欺诈概率: 金额 + 图特征(邻居欺诈数)
            fraud_neighbor_count = float(features[30])  # 第31维: fraud_neighbor_count
            graph_risk = min(0.3, fraud_neighbor_count / 10.0)
            
            if amount < 2000:
                amount_risk = 0.02
            elif amount < 10000:
                amount_risk = 0.08
            elif amount < 50000:
                amount_risk = 0.18
            else:
                amount_risk = 0.35
            
            base_prob = 0.01
            fraud_prob = min(0.95, base_prob + amount_risk + graph_risk)
            
            label = 1 if rng.random() < fraud_prob else 0
            
            X_list.append(features)
            y_list.append(label)
        
        db.close()
        
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int32)
        
        print(f"✅ 合成数据集大小: {X.shape}, 欺诈样本: {y.sum()}")
        
        # 分割为训练/验证/测试集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        fraud_ratio = y_train.mean()
        print(f"📊 训练集欺诈样本比例: {fraud_ratio*100:.3f}%")
        
        return fraud_ratio
    
    def handle_imbalance(self, use_smote=True):
        """处理样本不平衡问题"""
        if not use_smote:
            print("⚠️  不使用 SMOTE，直接使用原始数据")
            return
        
        print("🔄 使用 SMOTE 处理样本不平衡...")
        start_time = time.time()
        
        # SMOTE 过采样
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        elapsed = time.time() - start_time
        print(f"✅ SMOTE 完成 - 耗时: {elapsed:.2f}秒")
        print(f"   新训练集大小: {self.X_train.shape}")
        print(f"   正例: {self.y_train.sum()}, 负例: {len(self.y_train) - self.y_train.sum()}")
    
    def train_model(self, use_smote=True, use_gpu=False):
        """训练 XGBoost 模型"""
        print("\n🚀 开始训练 XGBoost 模型...")
        
        # 计算正负样本比例，用于 scale_pos_weight
        fraud_count = self.y_train.sum()
        normal_count = len(self.y_train) - fraud_count
        scale_pos_weight = normal_count / fraud_count if fraud_count > 0 else 1
        
        print(f"📊 scale_pos_weight: {scale_pos_weight:.2f}")
        
        # XGBoost 参数
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'scale_pos_weight': scale_pos_weight if not use_smote else 1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
        }
        
        # 创建模型
        self.model = xgb.XGBClassifier(**params)
        
        # 训练
        start_time = time.time()
        
        self.model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=10
        )
        
        elapsed = time.time() - start_time
        print(f"✅ 训练完成 - 耗时: {elapsed/60:.2f}分钟")
        
    def evaluate_model(self):
        """评估模型性能"""
        print("\n📊 评估模型性能...")
        
        # 在验证集上预测
        y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        # 尝试不同的阈值
        thresholds = [0.3, 0.5, 0.7]
        best_threshold = 0.5
        best_recall = 0
        
        print("\n🔍 测试不同阈值:")
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            print(f"\n  阈值 = {threshold:.1f}")
            print(f"    召回率: {recall*100:.2f}%")
            print(f"    精确率: {precision*100:.2f}%")
            print(f"    F1分数: {f1:.4f}")
            print(f"    误报率: {fpr*100:.2f}%")
            
            # 选择召回率最高且 >= 95% 的阈值
            if recall >= 0.95 and recall > best_recall:
                best_recall = recall
                best_threshold = threshold
        
        print(f"\n✅ 最佳阈值: {best_threshold} (召回率: {best_recall*100:.2f}%)")
        
        # 使用最佳阈值在测试集上评估
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_test_pred = (y_test_proba >= best_threshold).astype(int)
        
        print("\n📈 测试集最终性能:")
        print(classification_report(self.y_test, y_test_pred, 
                                   target_names=['正常', '欺诈']))
        
        # 计算 AUC
        auc = roc_auc_score(self.y_test, y_test_proba)
        print(f"\n🎯 AUC-ROC: {auc:.4f}")
        
        # 计算 Average Precision (PR-AUC)
        ap = average_precision_score(self.y_test, y_test_proba)
        print(f"🎯 Average Precision: {ap:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(f"\n混淆矩阵:")
        print(f"  TN: {cm[0,0]:6d}  |  FP: {cm[0,1]:6d}")
        print(f"  FN: {cm[1,0]:6d}  |  TP: {cm[1,1]:6d}")
        
        # 保存评估结果
        metrics = {
            'best_threshold': float(best_threshold),
            'auc': float(auc),
            'average_precision': float(ap),
            'test_metrics': {
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(
                    self.y_test, y_test_pred, 
                    target_names=['正常', '欺诈'],
                    output_dict=True
                )
            }
        }
        
        return metrics, best_threshold
    
    def save_model(self, metrics: dict, threshold: float):
        """保存模型和相关文件"""
        print("\n💾 保存模型...")
        
        # 保存 XGBoost 模型
        model_path = self.model_dir / "xgboost_model.json"
        self.model.save_model(model_path)
        print(f"✅ 模型已保存: {model_path}")
        
        # 保存评估指标
        metrics_path = self.model_dir / "model_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ 指标已保存: {metrics_path}")
        
        # 保存阈值配置
        config = {
            'threshold': threshold,
            'model_type': 'xgboost',
            'features_count': self.X_train.shape[1],
            'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_path = self.model_dir / "model_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ 配置已保存: {config_path}")
        
    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        print(f"\n📊 Top {top_n} 重要特征:")
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. V{idx:2d}: {importance[idx]:.4f}")


def main():
    """主函数"""
    print("=" * 60)
    print("🎯 XGBoost 反欺诈模型训练")
    print("=" * 60)
    
    # 路径配置
    data_dir = Path("data/processed")
    model_dir = Path("backend/models")
    
    # 创建训练器
    trainer = XGBoostTrainer(data_dir, model_dir)
    
    # 1. 生成 / 加载数据 (当前使用合成数据, 45维特征)
    # fraud_ratio = trainer.load_data()
    fraud_ratio = trainer.load_synthetic_data(n_samples=50000)
    
    # 2. 处理不平衡 (如果欺诈比例 < 1%，使用 SMOTE)
    use_smote = fraud_ratio < 0.01
    trainer.handle_imbalance(use_smote=use_smote)
    
    # 3. 训练模型
    trainer.train_model(use_smote=use_smote, use_gpu=False)
    
    # 4. 评估模型
    metrics, threshold = trainer.evaluate_model()
    
    # 5. 保存模型
    trainer.save_model(metrics, threshold)
    
    # 6. 特征重要性
    trainer.get_feature_importance()
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
