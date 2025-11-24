import argparse
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import schedule
import seaborn as sns
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from huggingface_hub import HfApi
from typing import List, Optional


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
        save_root: str = "stats/",
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
        
        top_n = 100
        top_datasets = self.processed_data.head(top_n).copy()
        
        plt.figure(figsize=(12, max(8, top_n * 0.3)))  # 动态调整高度
        
        # 创建横向条形图
        sns.barplot(
            data=top_datasets,
            y='dataset_name',
            x='downloads',
            orient='h',
            palette='viridis'
        )
        
        # 美化图表
        plt.title(f'Top {top_n} Most Downloaded {self.org_name} Datasets', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Download Count', fontsize=12)
        plt.ylabel('Dataset Name', fontsize=12)
        
        # 调整y轴标签显示
        y_labels = [name if len(name) < 40 else name[:37] + "..." 
                    for name in top_datasets['dataset_name']]
        plt.yticks(ticks=range(len(y_labels)), labels=y_labels)
        
        # 添加网格线
        plt.grid(True, alpha=0.3, axis='x')
        
        # 添加数据标签
        for i, (_, row) in enumerate(top_datasets.iterrows()):
            plt.text(
                row['downloads'] + max(top_datasets['downloads']) * 0.01,
                i,
                f"{row['downloads']:,}",
                va='center',
                fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root, "top_datasets_plot.png"), dpi=300, bbox_inches='tight')
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


def task(org_name, save_root, max_workers):
    save_root = os.path.join(save_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_root, exist_ok=True)
    analyzer = HuggingFaceDatasetAnalyzer(org_name, save_root, max_workers)
    analyzer.run()


def main(args):
    if args.now:
        task(args.org_name, args.save_root, args.max_workers)
    else:
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
    parser.add_argument("--save_root", type=str, default="stats/", help="Root directory to save stats")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of worker threads")
    parser.add_argument("--time", type=str, default="00:00", help="Time to run daily (HH:MM)")
    parser.add_argument("--now", action="store_true", help="Run the task immediately instead of scheduling")
    
    args = parser.parse_args()
    main(args)