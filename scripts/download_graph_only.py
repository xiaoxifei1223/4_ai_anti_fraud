"""
仅下载图数据集 (不需要 Kaggle API)
"""

import subprocess
import shutil
from pathlib import Path

# 设置目录
BASE_DIR = Path(__file__).parent.parent
GRAPH_DIR = BASE_DIR / "data" / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("📊 下载图网络数据集 (YelpChi + Amazon)")
print("="*60)

repo_url = "https://github.com/YingtongDou/CARE-GNN.git"
graph_repo_dir = GRAPH_DIR / "CARE-GNN"

print(f"\n🔗 GitHub 仓库: {repo_url}")
print(f"📁 目标目录: {graph_repo_dir}")

# 检查是否已存在
if graph_repo_dir.exists():
    print(f"\n⚠️  目录已存在")
    response = input("是否删除并重新下载? (y/n): ").strip().lower()
    if response == 'y':
        print("删除旧目录...")
        shutil.rmtree(graph_repo_dir)
    else:
        print("使用现有目录")
        exit(0)

# 开始下载
print("\n⏬ 开始 Clone 仓库...")
print("这可能需要几分钟，请耐心等待...\n")

try:
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(graph_repo_dir)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    if result.returncode == 0:
        print("\n✅ 下载成功!")
        
        # 检查数据文件
        data_dir = graph_repo_dir / "data"
        if data_dir.exists():
            print("\n📂 数据文件:")
            
            for dataset in ["yelp", "amazon"]:
                dataset_dir = data_dir / dataset
                if dataset_dir.exists():
                    print(f"\n  📊 {dataset.upper()} 数据集:")
                    files = list(dataset_dir.glob("*"))
                    total_size = 0
                    for f in files:
                        size = f.stat().st_size
                        total_size += size
                        size_mb = size / (1024 * 1024)
                        print(f"    ✓ {f.name} ({size_mb:.2f} MB)")
                    print(f"    总计: {total_size / (1024 * 1024):.2f} MB")
        
        print("\n" + "="*60)
        print("🎉 图数据集下载完成!")
        print("="*60)
        print(f"\n数据位置: {graph_repo_dir / 'data'}")
        
    else:
        print(f"\n❌ 下载失败!")
        print(f"错误信息: {result.stderr}")
        print("\n请检查:")
        print("1. 是否安装了 Git")
        print("2. 网络连接是否正常")
        
except FileNotFoundError:
    print("\n❌ 未找到 Git!")
    print("\n请先安装 Git:")
    print("访问: https://git-scm.com/downloads")
    
except Exception as e:
    print(f"\n❌ 发生错误: {e}")

print("\n" + "="*60)
