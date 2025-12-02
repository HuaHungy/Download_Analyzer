import argparse
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
try:
    import schedule
except Exception:
    schedule = None
import requests
import seaborn as sns
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from huggingface_hub import HfApi
from typing import List, Optional
from flask import Flask, request, jsonify
from flask import send_file, Response
DEFAULT_SAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "stats"))


@dataclass
class DatasetInfo:
    """数据容器类，用于存储数据集信息"""
    name: str
    downloads: int
    likes: int
    tags: List[str]
    success: bool


class HuggingFaceDatasetAnalyzer:
    """
    HuggingFace数据集分析器类
    封装数据获取、处理、可视化的完整功能
    """
    
    def __init__(
        self, 
        org_name: str = "huggingface", 
        save_root: str = DEFAULT_SAVE_ROOT,
        max_workers: int = 8,
    ):
        """
        初始化分析器
        
        Args:
            org_name: 组织名称
            max_workers: 最大工作线程数
        """
        self.org_name = org_name
        self.save_root = save_root
        self.max_workers = max_workers
        self.api = HfApi()
        self.datasets: List[DatasetInfo] = []
        self.dataframe: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
    
    def run(self) -> pd.DataFrame:
        """执行完整分析流程"""
        self.fetch_datasets_multithreaded()
        self.process_data()
        self.visualize_top_datasets()
        self.enhanced_analysis()
        self.process_summary()
        return self.processed_data if self.processed_data is not None else pd.DataFrame()
        
    def fetch_single_dataset(self, dataset_name: str) -> DatasetInfo:
        """
        获取单个数据集的详细信息
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            DatasetInfo: 数据集信息对象
        """
        try:
            dataset_info = self.api.dataset_info(dataset_name)
            downloads = getattr(dataset_info, 'downloads', 0)
            likes = getattr(dataset_info, 'likes', 0)
            tags = getattr(dataset_info, 'tags', [])
            
            return DatasetInfo(
                name=dataset_name,
                downloads=downloads,
                likes=likes,
                tags=tags,
                success=True
            )
        except Exception as e:
            print(f"Error fetching {dataset_name}: {e}")
            return DatasetInfo(
                name=dataset_name,
                downloads=0,
                likes=0,
                tags=[],
                success=False
            )
    
    def fetch_datasets_multithreaded(self) -> None:
        """
        使用多线程获取数据集信息
        
        Args:
            max_datasets: 最大数据集数量
        """
        print("Fetching dataset list...")
        datasets = list(self.api.list_datasets(author=self.org_name))
        
        print(f"Processing {len(datasets)} datasets with {self.max_workers} workers...")
        
        self.datasets = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dataset = {
                executor.submit(self.fetch_single_dataset, dataset.id): dataset.id 
                for dataset in datasets
            }
            
            for future in as_completed(future_to_dataset):
                dataset_name = future_to_dataset[future]
                try:
                    result = future.result()
                    self.datasets.append(result)
                    
                    if result.success and result.downloads > 0:
                        print(f"✓ {dataset_name}: {result.downloads} downloads")
                    elif not result.success:
                        print(f"✗ {dataset_name}: Failed to fetch")
                        
                except Exception as e:
                    print(f"Error processing {dataset_name}: {e}")
    
    def process_data(self) -> None:
        """处理获取的数据并转换为DataFrame"""
        if not self.datasets:
            raise ValueError("No data available.")
        
        # 转换为字典列表便于创建DataFrame
        data_dicts = []
        for dataset in self.datasets:
            data_dicts.append({
                'dataset_name': dataset.name,
                'downloads': dataset.downloads,
                'likes': dataset.likes,
                'tags': dataset.tags,
                'success': dataset.success
            })
        
        self.dataframe = pd.DataFrame(data_dicts)
        self.processed_data = self.dataframe[self.dataframe['success'] == True].copy()
        self.processed_data = self.processed_data.sort_values('downloads', ascending=False)
        
        print(f"Successfully processed {len(self.processed_data)} datasets")
        print(f"Total downloads: {self.processed_data['downloads'].sum():,}")
    
        save_path = os.path.join(self.save_root, "dataset_downloads.csv")
        self.processed_data.to_csv(save_path, index=False)
        print(f"Data saved to '{save_path}'")

    def visualize_top_datasets(self) -> None:
        """
        可视化前N个数据集
        
        Args:
            top_n: 显示前N个数据集
            save_path: 图表保存路径
        """
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        # 设置暗色主题科技风
        plt.style.use('dark_background')
        
        top_n = 100
        top_datasets = self.processed_data.head(top_n).copy()
        
        plt.figure(figsize=(14, max(12, top_n * 0.35)))  # 增加高度和宽度
        
        # 自定义颜色 - 青色/霓虹蓝
        bar_color = '#00f3ff' 
        
        # 创建横向条形图
        ax = sns.barplot(
            data=top_datasets,
            y='dataset_name',
            x='downloads',
            orient='h',
            color=bar_color,
            alpha=0.8
        )
        
        # 设置背景色
        ax.set_facecolor('#0b0c15') 
        plt.gcf().set_facecolor('#0b0c15')
        
        # 美化图表
        plt.title(f'HUGGINGFACE DOWNLOAD', 
                 fontsize=24, fontweight='bold', pad=25, color='#00f3ff', loc='left')
        plt.xlabel('DOWNLOAD COUNT', fontsize=16, color='#8892b0')
        plt.ylabel('DATASET NAME', fontsize=16, color='#8892b0')
        
        # 调整轴颜色
        ax.tick_params(axis='x', colors='#8892b0', labelsize=14)
        ax.tick_params(axis='y', colors='#8892b0', labelsize=14)
        ax.spines['bottom'].set_color('#00f3ff')
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#00f3ff')
        
        # 调整y轴标签显示
        y_labels = [name if len(name) < 40 else name[:37] + "..." 
                    for name in top_datasets['dataset_name']]
        plt.yticks(ticks=range(len(y_labels)), labels=y_labels)
        
        # 添加网格线 - 虚线，更有科技感
        plt.grid(True, alpha=0.1, axis='x', color='#00f3ff', linestyle='--')
        
        # 添加数据标签
        for i, (_, row) in enumerate(top_datasets.iterrows()):
            plt.text(
                row['downloads'] + max(top_datasets['downloads']) * 0.01,
                i,
                f"{row['downloads']:,}",
                va='center',
                fontsize=12,
                color='#ffffff',
                fontfamily='monospace'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root, "top_datasets_plot.png"), dpi=300, bbox_inches='tight', facecolor='#0b0c15')
        plt.close()
    
    def enhanced_analysis(self) -> None:
        """执行增强的数据分析"""
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        print("\n=== Enhanced Analysis ===")
        
        # 基本统计
        downloads = self.processed_data['downloads']
        print(f"Download statistics:")
        print(f"  - Mean: {downloads.mean():.0f}")
        print(f"  - Median: {downloads.median():.0f}")
        print(f"  - Max: {downloads.max():,}")
        print(f"  - Min: {downloads.min():,}")
        print(f"  - Std: {downloads.std():.0f}")
        
        # 标签分析（如果有标签数据）
        if 'tags' in self.processed_data.columns and len(self.processed_data) > 0:
            self._analyze_tags()
    
    def _analyze_tags(self) -> None:
        """分析数据集标签（内部方法）"""
        all_tags = []
        for tags in self.processed_data['tags']:
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        if all_tags:
            tag_counts = pd.Series(all_tags).value_counts().head(10)
            print("\nTop 10 most common tags:")
            for tag, count in tag_counts.items():
                print(f"  - {tag}: {count}")

    def process_summary(self) -> None:
        """获取分析摘要"""
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        print("\n=== Analysis Summary ===")
        print(f"Total datasets processed: {len(self.datasets)}")
        print(f"Successful fetches: {len(self.processed_data)}")
        print(f"Top dataset: {self.processed_data.iloc[0]['dataset_name']} "
              f"with {self.processed_data.iloc[0]['downloads']:,} downloads")
        print(f"Total downloads: {self.processed_data['downloads'].sum():,}")

        summary_df = pd.DataFrame({
            'Total Datasets': [len(self.datasets)],
            'Successful Fetches': [len(self.processed_data)],
            'Total Downloads': [self.processed_data['downloads'].sum()],
            'Mean Downloads': [self.processed_data['downloads'].mean()],
        })

        save_path = os.path.join(self.save_root, "analysis_summary.csv")
        summary_df.to_csv(save_path, index=False)
        print(f"Summary saved to '{save_path}'")


class ModelScopeDatasetAnalyzer:
    def __init__(self, org_name: str = "RoboCOIN", save_root: str = DEFAULT_SAVE_ROOT):
        self.org_name = org_name
        self.save_root = save_root
        self.datasets = []
        self.dataframe: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        self.fetch_datasets()
        self.process_data()
        self.save_csv()
        return self.dataframe if self.dataframe is not None else pd.DataFrame()

    def fetch_datasets(self) -> None:
        base_url = "https://modelscope.cn/api/v1/datasets"
        params = {"owner": self.org_name, "page": 1, "PageSize": 1000}
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"ModelScope HTTP {resp.status_code}")
                self.datasets = []
                return
            data = resp.json()
            if data.get("Code") != 200:
                print(f"ModelScope API error: {data}")
                self.datasets = []
                return
            for d in data.get("Data", []):
                name = f"{d.get('Namespace')}/{d.get('Name')}"
                downloads = d.get("Downloads", 0)
                self.datasets.append({
                    "dataset_name": name,
                    "downloads": downloads,
                    "likes": 0,
                    "tags": [],
                    "success": True,
                })
        except Exception as e:
            print(f"ModelScope fetch error: {e}")
            self.datasets = []

    def process_data(self) -> None:
        if not self.datasets:
            self.dataframe = pd.DataFrame(columns=["dataset_name","downloads","likes","tags","success"])
            return
        df = pd.DataFrame(self.datasets)
        df = df[df['success'] == True].copy()
        self.dataframe = df.sort_values('downloads', ascending=False)

    def save_csv(self) -> None:
        try:
            save_path = os.path.join(self.save_root, "modelscope_dataset_downloads.csv")
            if self.dataframe is not None:
                self.dataframe.to_csv(save_path, index=False)
        except Exception:
            pass

    def visualize_top_datasets(self) -> None:
        if self.dataframe is None or self.dataframe.empty:
            return
        plt.style.use('dark_background')
        top_n = 100
        top_datasets = self.dataframe.head(top_n).copy()
        plt.figure(figsize=(14, max(12, top_n * 0.35)))
        bar_color = '#00f3ff'
        ax = sns.barplot(
            data=top_datasets,
            y='dataset_name',
            x='downloads',
            orient='h',
            color=bar_color,
            alpha=0.8
        )
        ax.set_facecolor('#0b0c15')
        plt.gcf().set_facecolor('#0b0c15')
        plt.title(f'MODELSCOPE DOWNLOAD', 
                 fontsize=24, fontweight='bold', pad=25, color='#00f3ff', loc='left')
        plt.xlabel('DOWNLOAD COUNT', fontsize=16, color='#8892b0')
        plt.ylabel('DATASET NAME', fontsize=16, color='#8892b0')
        ax.tick_params(axis='x', colors='#8892b0', labelsize=14)
        ax.tick_params(axis='y', colors='#8892b0', labelsize=14)
        ax.spines['bottom'].set_color('#00f3ff')
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#00f3ff')
        y_labels = [name if len(name) < 40 else name[:37] + "..." for name in top_datasets['dataset_name']]
        plt.yticks(ticks=range(len(y_labels)), labels=y_labels)
        plt.grid(True, alpha=0.1, axis='x', color='#00f3ff', linestyle='--')
        for i, (_, row) in enumerate(top_datasets.iterrows()):
            plt.text(
                row['downloads'] + max(top_datasets['downloads']) * 0.01,
                i,
                f"{row['downloads']:,}",
                va='center',
                fontsize=12,
                color='#ffffff',
                fontfamily='monospace'
            )
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root, "modelscope_top_datasets_plot.png"), dpi=300, bbox_inches='tight', facecolor='#0b0c15')
        plt.close()


def task(org_name, save_root, max_workers):
    save_root = os.path.join(save_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_root, exist_ok=True)
    analyzer = HuggingFaceDatasetAnalyzer(org_name, save_root, max_workers)
    return analyzer.run()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/analyze")
    def analyze():
        data = request.get_json(silent=True) or {}
        org_name = data.get("org_name", request.args.get("org_name", "huggingface"))
        save_root = data.get("save_root", request.args.get("save_root", DEFAULT_SAVE_ROOT))
        try:
            max_workers = int(data.get("max_workers", request.args.get("max_workers", 8)))
        except Exception:
            max_workers = 8

        save_path_root = os.path.join(save_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_path_root, exist_ok=True)
        analyzer = HuggingFaceDatasetAnalyzer(org_name, save_path_root, max_workers)
        df = analyzer.run()
        ms_org = data.get("ms_org_name", request.args.get("ms_org_name", org_name))
        ms_analyzer = ModelScopeDatasetAnalyzer(ms_org, save_path_root)
        ms_df = ms_analyzer.run()
        hf_total = int(df["downloads"].sum()) if df is not None and not df.empty else 0
        ms_total = int(ms_df["downloads"].sum()) if ms_df is not None and not ms_df.empty else 0
        summary = {
            "org_name": org_name,
            "modelscope_org": ms_org,
            "total_datasets": len(analyzer.datasets),
            "successful_fetches": len(df) if df is not None else 0,
            "total_downloads": hf_total,
            "modelscope_total_downloads": ms_total,
            "combined_total_downloads": hf_total + ms_total,
            "save_dir": save_path_root,
            "csv_path": os.path.join(save_path_root, "dataset_downloads.csv"),
            "ms_csv_path": os.path.join(save_path_root, "modelscope_dataset_downloads.csv"),
            "summary_path": os.path.join(save_path_root, "analysis_summary.csv"),
            "plot_path": os.path.join(save_path_root, "top_datasets_plot.png"),
        }
        return jsonify(summary)

    @app.get("/")
    def view():
        org_name = "RoboCOIN"
        save_root = request.args.get("save_root", DEFAULT_SAVE_ROOT)
        try:
            max_workers = int(request.args.get("max_workers", 8))
        except Exception:
            max_workers = 8

        save_path_root = os.path.join(save_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_path_root, exist_ok=True)
        analyzer = HuggingFaceDatasetAnalyzer(org_name, save_path_root, max_workers)
        df = analyzer.run()
        csv_path = os.path.join(save_path_root, "dataset_downloads.csv")
        plot_path = os.path.join(save_path_root, "top_datasets_plot.png")
        if not os.path.isabs(plot_path):
            plot_path = os.path.abspath(plot_path)
        total_dl = int(df["downloads"].sum()) if df is not None and not df.empty else 0
        ms_org = request.args.get("ms_org_name", org_name)
        ms_analyzer = ModelScopeDatasetAnalyzer(ms_org, save_path_root)
        ms_df = ms_analyzer.run()
        ms_analyzer.visualize_top_datasets()
        ms_total = int(ms_df["downloads"].sum()) if ms_df is not None and not ms_df.empty else 0
        combined_total = total_dl + ms_total
        ms_plot_path = os.path.join(save_path_root, "modelscope_top_datasets_plot.png")
        favicon_path = request.args.get("favicon", request.args.get("favicon_path"))
        favicon_tag = f'<link rel="icon" href="/static_favicon?path={favicon_path}" />' if favicon_path else ""
        _ico = "/home/kemove/Analyzer/assert/logo.ico"
        _png = "/home/kemove/Analyzer/assert/logo.png"
        _fav = _ico if os.path.exists(_ico) else (_png if os.path.exists(_png) else None)
        _ver = str(int(os.path.getmtime(_fav))) if _fav else str(int(time.time()))
        default_favicon_tag = (
            f'<link rel="icon" type="image/x-icon" href="/favicon.ico?v={_ver}" />'
            if (_fav and _fav.endswith(".ico")) else
            f'<link rel="icon" type="image/png" href="/favicon.ico?v={_ver}" />'
        )

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HF ANALYZER // {org_name.upper()}</title>
    {default_favicon_tag}
    {favicon_tag}
    <style>
        :root {{
            --bg-color: #050608;
            --card-bg: rgba(21, 22, 33, 0.7);
            --primary-color: #00f3ff;
            --secondary-color: #7000ff;
            --text-main: #e6f1ff;
            --text-dim: #aab4d0;
            --border-color: #233554;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'SF Mono', 'Fira Code', 'Roboto Mono', monospace;
            margin: 0;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
            min-height: 100vh;
            background-image:
                radial-gradient(ellipse at 20% 30%, rgba(92,0,255,0.18) 0%, rgba(92,0,255,0.0) 35%),
                radial-gradient(ellipse at 70% 70%, rgba(0,120,255,0.14) 0%, rgba(0,120,255,0.0) 40%),
                radial-gradient(ellipse at 30% 85%, rgba(0,200,170,0.10) 0%, rgba(0,200,170,0.0) 35%);
            background-attachment: fixed;
        }}

        /* deep-space nebula + dynamic star layers */
        .nebula {{
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background:
                radial-gradient(circle at 20% 30%, rgba(92,0,255,0.20) 0%, rgba(92,0,255,0.0) 40%),
                radial-gradient(circle at 70% 60%, rgba(0,120,255,0.16) 0%, rgba(0,120,255,0.0) 42%),
                radial-gradient(circle at 30% 80%, rgba(0,200,170,0.12) 0%, rgba(0,200,170,0.0) 38%);
            filter: blur(45px);
            animation: nebulaShift 120s ease-in-out infinite alternate;
            z-index: -3;
        }}
        @keyframes nebulaShift {{
            0% {{ transform: translate3d(0,0,0) scale(1); filter: hue-rotate(0deg); }}
            100% {{ transform: translate3d(20px,-20px,0) scale(1.05); filter: hue-rotate(30deg); }}
        }}

        /* dynamic star layers */
        .stars, .stars2, .stars3 {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            background-repeat: repeat;
            z-index: -1;
            opacity: 0.7;
        }}
        .stars {{
            background-image:
                radial-gradient(circle, rgba(180,225,255,0.85) 1px, transparent 3px);
            background-size: 300px 300px;
            animation: drift1 120s linear infinite, twinkle 3s ease-in-out infinite;
            mix-blend-mode: screen;
            filter: blur(0.2px);
        }}
        .stars2 {{
            background-image:
                radial-gradient(circle, rgba(120,180,255,0.7) 1.2px, transparent 3px);
            background-size: 450px 450px;
            animation: drift2 180s linear infinite, twinkle 4s ease-in-out infinite;
            opacity: 0.5;
            mix-blend-mode: screen;
            filter: blur(0.3px);
        }}
        .stars3 {{
            background-image:
                radial-gradient(circle, rgba(160,100,255,0.6) 1.6px, transparent 4px);
            background-size: 700px 700px;
            animation: drift3 240s linear infinite, twinkle 5s ease-in-out infinite;
            opacity: 0.35;
            mix-blend-mode: screen;
            filter: blur(0.35px);
        }}

        @keyframes drift1 {{
            0% {{ transform: translate3d(0,0,0); }}
            100% {{ transform: translate3d(-50px,-50px,0); }}
        }}
        @keyframes drift2 {{
            0% {{ transform: translate3d(0,0,0); }}
            100% {{ transform: translate3d(50px,-30px,0); }}
        }}
        @keyframes drift3 {{
            0% {{ transform: translate3d(0,0,0); }}
            100% {{ transform: translate3d(-30px,50px,0); }}
        }}
        @keyframes twinkle {{
            0%, 100% {{ filter: brightness(1); }}
            50% {{ filter: brightness(1.6); }}
        }}

        /* subtle grid overlay for tech feel */
        .grid-overlay {{
            position: fixed;
            top:0;left:0;width:100%;height:100%;
            background-image:
                linear-gradient(rgba(0, 243, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 243, 255, 0.02) 1px, transparent 1px);
            background-size: 60px 60px, 60px 60px;
            z-index: -2;
            opacity: 0.08;
            mix-blend-mode: screen;
        }}

        /* container sits above star layers */
        .container {{ position: relative; z-index: 1; }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            position: relative;
        }}
        
        header::after {{
            content: '';
            position: absolute;
            bottom: -6px;
            left: 0;
            width: 100px;
            height: 10px;
            background: var(--primary-color);
        }}

        h1 {{
            margin: 0;
            font-size: 4rem;
            text-transform: uppercase;
            letter-spacing: 4px;
            text-shadow: 0 0 20px rgba(0, 243, 255, 0.6);
        }}

        .subtitle {{
            color: var(--primary-color);
            font-size: 1.5rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 20px;
            position: relative;
            overflow: hidden;
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 2px;
            height: 100%;
            background: var(--secondary-color);
        }}

        .stat-label {{
            color: var(--text-dim);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}

        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--text-main);
            text-shadow: 0 0 10px rgba(230, 241, 255, 0.3);
        }}

        .main-content {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 28px;
        }}

        @media (max-width: 768px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
        }}

        .panel {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}

        .panel-title {{
            font-size: 1.8rem;
            color: var(--primary-color);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        img.plot-img {{
            width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
            filter: drop-shadow(0 0 5px rgba(0,0,0,0.5));
        }}

        .action-btn {{
            display: inline-block;
            background: rgba(0, 243, 255, 0.1);
            color: var(--primary-color);
            text-decoration: none;
            padding: 10px 20px;
            border: 1px solid var(--primary-color);
            text-transform: uppercase;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }}

        .action-btn:hover {{
            background: var(--primary-color);
            color: #000;
            box-shadow: 0 0 15px var(--primary-color);
        }}
        
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: var(--text-dim);
            font-size: 0.8rem;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }}
        
        /* Blink animation for 'LIVE' status or similar */
        @keyframes blink {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        .status-dot {{
            display: inline-block;
            width: 10px;
            height: 10px;
            background-color: var(--primary-color);
            border-radius: 50%;
            margin-right: 8px;
            box-shadow: 0 0 5px var(--primary-color);
            animation: blink 2s infinite;
        }}
    </style>
</head>
<body>
    <div class="nebula"></div>
    <div class="stars"></div>
    <div class="stars2"></div>
    <div class="stars3"></div>
    <div class="grid-overlay"></div>
    <div class="container">
        <header>
            <div>
                <div class="subtitle">SYSTEM: ANALYZER HF_ORG: {org_name} MS_ORG: {ms_org}</div>
                <h1>{org_name} Dataset Analyzer</h1>
            </div>
            <div style="text-align: right;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div style="color: var(--primary-color);"><span class="status-dot"></span>SYSTEM ONLINE</div>
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">HuggingFace Downloads</div>
                <div class="stat-value">{total_dl:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">ModelScope Downloads</div>
                <div class="stat-value">{ms_total:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Downloads</div>
                <div class="stat-value">{combined_total:,}</div>
            </div>
             <div class="stat-card">
                <div class="stat-label">Status</div>
                <div class="stat-value" style="color: #00ff9d;">COMPLETED</div>
            </div>
        </div>

        <div class="main-content">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">>> HuggingFace Download Statistics Visualization</div>
                </div>
                <img src="/static_plot?path={plot_path}" class="plot-img" alt="HF Top Datasets Plot">
            </div>
            
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">>> ModelScope Download Statistics Visualization</div>
                </div>
                <img src="/static_plot?path={ms_plot_path}" class="plot-img" alt="ModelScope Top Datasets Plot">
            </div>
            
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">>> Data Export</div>
                </div>
                <p style="color: var(--text-dim); margin-bottom: 20px; font-size: 1rem;">
                    Export CSV for both platforms.
                </p>
                <div style="display:flex; gap:20px; justify-content:center;">
                    <a href="/static_csv?path={csv_path}" class="action-btn" style="font-size:1rem; padding:12px 24px;">[ DOWNLOAD HF CSV ]</a>
                    <a href="/static_csv?path={os.path.join(save_path_root, 'modelscope_dataset_downloads.csv')}" class="action-btn" style="font-size:1rem; padding:12px 24px;">[ DOWNLOAD MS CSV ]</a>
                </div>
                
                <div style="margin-top: 40px; border-top: 1px dashed var(--border-color); padding-top: 20px;">
                    <div class="panel-title" style="font-size: 0.9rem; margin-bottom: 10px;">SYSTEM LOGS</div>
                    <div style="font-family: monospace; font-size: 0.8rem; color: var(--text-dim);">
                        > Initializing analysis... OK<br>
                        > Fetching {org_name}... OK<br>
                        > Processing data... OK<br>
                        > Rendering visualization... OK
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            HUGGINGFACE DATASET ANALYZER v2.0 | POWERED BY FLASK | TERMINAL UI
        </div>
    </div>
</body>
</html>
"""
        return Response(html, mimetype="text/html")

    @app.get("/static_plot")
    def static_plot():
        path = request.args.get("path")
        if not path or not os.path.exists(path):
            return jsonify({"error": "plot not found"}), 404
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return send_file(path, mimetype="image/png")

    @app.get("/static_csv")
    def static_csv():
        path = request.args.get("path")
        if not path or not os.path.exists(path):
            return jsonify({"error": "csv not found"}), 404
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return send_file(path, mimetype="text/csv", as_attachment=True, download_name=os.path.basename(path))

    @app.get("/favicon.ico")
    def favicon():
        ico = "/home/kemove/Analyzer/assert/logo.ico"
        png = "/home/kemove/Analyzer/assert/logo.png"
        path = ico if os.path.exists(ico) else (png if os.path.exists(png) else None)
        if path:
            ext = os.path.splitext(path)[1].lower()
            mime = "image/x-icon" if ext == ".ico" else "image/png"
            return send_file(path, mimetype=mime)
        return jsonify({"error": "favicon not found"}), 404

    @app.get("/static_favicon")
    def static_favicon():
        path = request.args.get("path")
        if not path or not os.path.exists(path):
            return jsonify({"error": "favicon not found"}), 404
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        ext = os.path.splitext(path)[1].lower()
        mime = "image/x-icon" if ext == ".ico" else ("image/png" if ext == ".png" else "image/jpeg")
        return send_file(path, mimetype=mime)


    return app

def main(args):
    if args.now:
        task(args.org_name, args.save_root, args.max_workers)
    elif os.getenv("HF_ANALYZER_FLASK", "0") == "1":
        app = create_app()
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
    else:
        if schedule is None:
            print("The 'schedule' package is not installed. Install with: pip install schedule")
            return
        schedule.every().day.at(args.time).do(
            task, org_name=args.org_name, save_root=args.save_root, max_workers=args.max_workers
        )
        print("Scheduler started. Waiting for the next run...")
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace Dataset Analyzer Scheduler")
    parser.add_argument("--org_name", type=str, default="RoboCOIN", help="Organization name")
    parser.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT, help="Root directory to save stats")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of worker threads")
    parser.add_argument("--time", type=str, default="00:00", help="Time to run daily (HH:MM)")
    parser.add_argument("--now", action="store_true", help="Run the task immediately instead of scheduling")
    
    args = parser.parse_args()
    main(args)
