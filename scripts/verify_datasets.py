"""
验证数据集下载情况
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

def check_file(file_path: Path, name: str) -> bool:
    """检查文件是否存在"""
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {name}")
        print(f"     📁 {file_path.relative_to(BASE_DIR)}")
        print(f"     📊 大小: {size_mb:.2f} MB")
        return True
    else:
        print(f"  ❌ {name}")
        print(f"     📁 期望位置: {file_path.relative_to(BASE_DIR)}")
        return False

def main():
    print("="*60)
    print("📊 数据集下载验证")
    print("="*60)
    
    results = {}
    
    # 1. Kaggle Credit Card
    print("\n1️⃣  Kaggle Credit Card Fraud Detection")
    print("-" * 60)
    creditcard_file = DATA_DIR / "raw" / "creditcard" / "creditcard.csv"
    results['creditcard'] = check_file(creditcard_file, "creditcard.csv")
    
    if not results['creditcard']:
        print("     ⚠️  需要手动下载")
        print("     🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"     📥 下载后放到: {creditcard_file.parent}")
    
    # 2. YelpChi
    print("\n2️⃣  YelpChi Graph Dataset")
    print("-" * 60)
    yelp_file = DATA_DIR / "graph" / "CARE-GNN" / "data" / "yelp" / "YelpChi.mat"
    results['yelp'] = check_file(yelp_file, "YelpChi.mat")
    
    # 3. Amazon
    print("\n3️⃣  Amazon Graph Dataset")
    print("-" * 60)
    amazon_file = DATA_DIR / "graph" / "CARE-GNN" / "data" / "amazon" / "Amazon.mat"
    results['amazon'] = check_file(amazon_file, "Amazon.mat")
    
    # 总结
    print("\n" + "="*60)
    print("📈 验证总结")
    print("="*60)
    
    total = len(results)
    success = sum(results.values())
    
    print(f"\n已下载: {success}/{total} 个数据集")
    
    for name, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {name}")
    
    if success == total:
        print("\n🎉 所有数据集已就绪!")
        print("\n下一步:")
        print("  1. 运行数据预处理: python scripts/preprocess_data.py")
        print("  2. 探索数据: python scripts/analyze_data.py")
        return 0
    else:
        print("\n⚠️  还有数据集未下载")
        print("\n请查看: data/下载指南.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
