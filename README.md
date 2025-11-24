# HuggingFace数据集分析器

一个基于Python的自动化工具，用于分析和可视化HuggingFace组织下的数据集下载情况。该工具支持多线程数据获取、自动可视化生成、定期调度运行等功能。

## 功能特点

- 基于Hugginface API的多线程数据获取
- 数据分析与可视化
- 任务自动定时调度运行

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用（立即运行）

```bash
python analyzer.py --now
```

### 定时任务模式

```bash
# 每天00:00自动运行
python analyzer.py --time "00:00"

# 自定义组织和分析时间
python analyzer.py --org_name "RoboCOIN" --time "09:00" --max_workers 12
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|------|--------|
| --org_name | str | "RoboCOIN" | 要分析的HuggingFace组织名 |
| --save_root | str | "stats/" | 结果保存根目录 |
| --max_workers | int | 8 | 最大工作线程数 |
| --time | str | "00:00" | 每日运行时间（HH:MM） |
| --now | flag | False | 立即运行而不调度 |

### 输出结果

工具运行后会在指定目录生成以下文件：

```bash
stats/
└── YYYYMMDD_HHMMSS/          # 时间戳目录
    ├── dataset_downloads.csv  # 完整数据集下载数据
    ├── analysis_summary.csv  # 分析摘要统计
    └── top_datasets_plot.png # Top 100数据集可视化图表
```

### 示例输出

```bash
Fetching dataset list...
Processing 150 datasets with 8 workers...
✓ dataset1: 15000 downloads
✓ dataset2: 12000 downloads
...
Successfully processed 145 datasets
Total downloads: 1,234,567

=== Enhanced Analysis ===
Download statistics:
  - Mean: 8,512
  - Median: 1,234
  - Max: 150,000
  - Min: 0
  - Std: 12,345

=== Analysis Summary ===
Total datasets processed: 150
Successful fetches: 145
Top dataset: example-dataset with 150,000 downloads
Total downloads: 1,234,567
```