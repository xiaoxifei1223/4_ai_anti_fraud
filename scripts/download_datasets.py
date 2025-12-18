"""
数据集下载脚本
下载反欺诈系统所需的公开数据集
"""

import os
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path
import sys

# 设置数据目录
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GRAPH_DIR = DATA_DIR / "graph"

# 确保目录存在
RAW_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest_path: Path, desc: str = ""):
    """下载文件并显示进度"""
    print(f"\n{'='*60}")
    print(f"📥 下载: {desc}")
    print(f"URL: {url}")
    print(f"目标: {dest_path}")
    print(f"{'='*60}")
    
    try:
        def reporthook(count, block_size, total_size):
            """进度回调"""
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                size_mb = total_size / (1024 * 1024)
                downloaded_mb = count * block_size / (1024 * 1024)
                sys.stdout.write(f"\r进度: {percent}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n✅ 下载完成: {dest_path.name}")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """解压 ZIP 文件"""
    print(f"\n📦 解压: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ 解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False


def extract_gz(gz_path: Path, extract_to: Path):
    """解压 GZ 文件"""
    print(f"\n📦 解压: {gz_path.name}")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(extract_to, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✅ 解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False


def download_kaggle_creditcard():
    """
    下载 Kaggle Credit Card Fraud Detection 数据集
    注意: 需要先配置 Kaggle API (kaggle.json)
    """
    print("\n" + "="*60)
    print("📊 数据集 1: Kaggle Credit Card Fraud Detection")
    print("="*60)
    
    dataset_name = "mlg-ulb/creditcardfraud"
    output_dir = RAW_DIR / "creditcard"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n使用 Kaggle API 下载: {dataset_name}")
    print(f"目标目录: {output_dir}")
    
    try:
        # 检查是否安装了 kaggle
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("\n开始下载...")
        api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        
        print(f"\n✅ Kaggle Credit Card 数据集下载成功!")
        print(f"📁 文件位置: {output_dir}")
        
        # 列出下载的文件
        files = list(output_dir.glob("*"))
        if files:
            print("\n下载的文件:")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")
        
        return True
        
    except ImportError:
        print("\n⚠️  未安装 Kaggle API")
        print("请运行: pip install kaggle")
        print("\n然后配置 API Token:")
        print("1. 访问 https://www.kaggle.com/settings")
        print("2. 创建新的 API Token")
        print("3. 下载 kaggle.json 到 ~/.kaggle/")
        return False
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n备用方案: 手动下载")
        print(f"1. 访问: https://www.kaggle.com/datasets/{dataset_name}")
        print("2. 下载 ZIP 文件")
        print(f"3. 解压到: {output_dir}")
        return False


def download_graph_datasets():
    """
    下载图网络数据集 (YelpChi, Amazon)
    从 GitHub: YingtongDou/CARE-GNN
    """
    print("\n" + "="*60)
    print("📊 数据集 2-3: YelpChi + Amazon Fraud Graph Datasets")
    print("="*60)
    
    repo_url = "https://github.com/YingtongDou/CARE-GNN.git"
    graph_repo_dir = GRAPH_DIR / "CARE-GNN"
    
    print(f"\nClone GitHub 仓库: {repo_url}")
    print(f"目标目录: {graph_repo_dir}")
    
    try:
        import subprocess
        
        # 检查目录是否已存在
        if graph_repo_dir.exists():
            print(f"\n⚠️  目录已存在: {graph_repo_dir}")
            user_input = input("是否重新下载? (y/n): ").strip().lower()
            if user_input == 'y':
                shutil.rmtree(graph_repo_dir)
            else:
                print("跳过下载")
                return True
        
        # Clone 仓库
        print("\n开始 Clone...")
        result = subprocess.run(
            ["git", "clone", repo_url, str(graph_repo_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ 图数据集下载成功!")
            print(f"📁 位置: {graph_repo_dir}")
            
            # 检查数据文件
            data_dir = graph_repo_dir / "data"
            if data_dir.exists():
                print("\n数据文件:")
                for dataset in ["yelp", "amazon"]:
                    dataset_dir = data_dir / dataset
                    if dataset_dir.exists():
                        print(f"\n  📂 {dataset.upper()}:")
                        for f in dataset_dir.glob("*"):
                            size_mb = f.stat().st_size / (1024 * 1024)
                            print(f"    - {f.name} ({size_mb:.2f} MB)")
            
            return True
        else:
            print(f"\n❌ Clone 失败: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("\n❌ 未找到 Git 命令")
        print("请安装 Git: https://git-scm.com/downloads")
        print("\n备用方案: 手动下载")
        print(f"1. 访问: {repo_url}")
        print("2. 下载 ZIP")
        print(f"3. 解压到: {graph_repo_dir}")
        return False
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def download_ieee_cis():
    """
    下载 IEEE-CIS Fraud Detection 数据集
    这是一个 Kaggle 竞赛数据集，需要同意竞赛规则
    """
    print("\n" + "="*60)
    print("📊 数据集 4: IEEE-CIS Fraud Detection (可选)")
    print("="*60)
    
    competition_name = "ieee-fraud-detection"
    output_dir = RAW_DIR / "ieee_cis"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n使用 Kaggle API 下载竞赛: {competition_name}")
    print(f"目标目录: {output_dir}")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # 检查是否已接受竞赛规则
        print("\n⚠️  注意: 此数据集来自 Kaggle 竞赛")
        print("需要先访问竞赛页面并同意规则:")
        print(f"https://www.kaggle.com/c/{competition_name}")
        
        user_input = input("\n已同意规则? (y/n): ").strip().lower()
        if user_input != 'y':
            print("跳过下载")
            return False
        
        print("\n开始下载...")
        api.competition_download_files(competition_name, path=output_dir)
        
        # 解压所有 zip 文件
        for zip_file in output_dir.glob("*.zip"):
            extract_zip(zip_file, output_dir)
            zip_file.unlink()  # 删除 zip 文件
        
        print(f"\n✅ IEEE-CIS 数据集下载成功!")
        print(f"📁 位置: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n这个数据集较大 (~500MB)，如果不需要可以跳过")
        return False


def create_readme():
    """创建数据集说明文件"""
    readme_content = """# 反欺诈数据集

## 已下载的数据集

### 1. Kaggle Credit Card Fraud Detection
- **位置**: `data/raw/creditcard/`
- **文件**: `creditcard.csv`
- **规模**: 284,807 笔交易
- **欺诈率**: 0.172%
- **特征**: 30 列 (Time, V1-V28, Amount, Class)
- **用途**: 测试快速反欺诈模块 (XGBoost/LightGBM)

### 2. YelpChi Fraud Dataset
- **位置**: `data/graph/CARE-GNN/data/yelp/`
- **规模**: 45,954 个用户节点
- **欺诈率**: 8.37%
- **图结构**: 用户-评论-商家关系图
- **用途**: 测试图网络分析 (Neo4j + GNN)

### 3. Amazon Fraud Dataset
- **位置**: `data/graph/CARE-GNN/data/amazon/`
- **规模**: 11,944 个节点
- **欺诈率**: 9.5%
- **图结构**: 多关系异构图
- **用途**: 测试团伙欺诈识别

### 4. IEEE-CIS Fraud Detection (可选)
- **位置**: `data/raw/ieee_cis/`
- **规模**: 590,540 笔交易
- **特征**: 394 列
- **用途**: 完整系统测试

## 数据预处理

运行以下脚本进行数据预处理:
```bash
python scripts/preprocess_data.py
```

## 数据目录结构
```
data/
├── raw/                    # 原始数据
│   ├── creditcard/
│   └── ieee_cis/
├── processed/              # 预处理后的数据
│   ├── train/
│   ├── val/
│   └── test/
└── graph/                  # 图数据
    └── CARE-GNN/
```

## 下载新数据集

运行下载脚本:
```bash
python scripts/download_datasets.py
```
"""
    
    readme_path = DATA_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✅ 创建数据集说明: {readme_path}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 反欺诈系统 - 数据集下载工具")
    print("="*60)
    
    print(f"\n数据目录: {DATA_DIR}")
    print(f"  - 原始数据: {RAW_DIR}")
    print(f"  - 图数据: {GRAPH_DIR}")
    
    # 下载列表
    datasets = [
        ("Kaggle Credit Card Fraud", download_kaggle_creditcard),
        ("Graph Datasets (Yelp + Amazon)", download_graph_datasets),
    ]
    
    # 询问是否下载可选数据集
    print("\n" + "="*60)
    print("可选数据集:")
    print("  - IEEE-CIS Fraud Detection (~500MB, 需同意竞赛规则)")
    user_input = input("\n是否下载可选数据集? (y/n): ").strip().lower()
    if user_input == 'y':
        datasets.append(("IEEE-CIS Fraud Detection", download_ieee_cis))
    
    # 执行下载
    results = {}
    for name, download_func in datasets:
        success = download_func()
        results[name] = success
    
    # 创建说明文件
    create_readme()
    
    # 总结
    print("\n" + "="*60)
    print("📊 下载总结")
    print("="*60)
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{status}: {name}")
    
    # 下一步提示
    print("\n" + "="*60)
    print("🎯 下一步操作")
    print("="*60)
    
    if any(results.values()):
        print("\n已下载数据集，接下来可以:")
        print("1. 运行数据预处理: python scripts/preprocess_data.py")
        print("2. 查看数据分析: python scripts/analyze_data.py")
        print("3. 训练模型: python scripts/train_model.py")
        print(f"\n数据集说明: {DATA_DIR / 'README.md'}")
    else:
        print("\n⚠️  所有数据集下载失败")
        print("请检查:")
        print("1. 网络连接")
        print("2. Kaggle API 配置 (pip install kaggle)")
        print("3. Git 安装")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
