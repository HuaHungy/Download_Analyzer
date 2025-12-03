# Analyzer 项目说明

分析 HuggingFace 与 ModelScope 数据集下载量，生成 CSV 与可视化图表，并提供 Web/API 入口。核心文件：`analyzer.py`、`hf_analyzer_local.py`、`ms_analyzer_local.py`。

## 安装依赖

```bash
pip install -r requirements.txt
```

## analyzer.py（Web/API + CLI/定时）

- 功能
  - 抓取 HuggingFace 组织数据集下载量（优先使用 list_datasets 的聚合信息；缺失时并发回退到 dataset_info）
  - 生成 `huggingface_dataset_downloads.csv` 与 `huggingface_top_datasets_plot.png`
  - 可选抓取并对比 ModelScope，生成 `modelscope_dataset_downloads.csv` 与图表
  - 提供 Web 页面与 API 接口；支持自定义 favicon 与静态文件下载

- 启动 Web
```bash
HF_ANALYZER_FLASK=1 PORT=8000 python /home/kemove/Analyzer/analyzer.py
```
访问 `http://localhost:8000/`；URL 支持：`ms_org_name`、`save_root`、`max_workers`、`favicon`。

- 常用接口
  - `GET /health`
  - `GET /` 渲染页面并执行分析（生成 CSV/图表）
  - `POST /analyze` 触发分析并返回 JSON 摘要
  - `GET /static_plot?path=<abs_path>` 返回 PNG 图表
  - `GET /static_csv?path=<abs_path>` 下载 CSV
  - `GET /static_favicon?path=<abs_path>` 返回图标
  - `GET /favicon.ico` 默认网站图标（支持 PNG 回退）

- 自定义图标
  - 环境变量：`FAVICON_PATH=/home/kemove/Analyzer/assert/logo.ico`
  - URL 参数：`/?favicon=/home/kemove/Analyzer/assert/logo.png`

- CLI 示例
```bash
# 立即执行一次
python /home/kemove/Analyzer/analyzer.py --now --org_name RoboCOIN --max_workers 32

# 每日定时（需安装 schedule）
python /home/kemove/Analyzer/analyzer.py --time "00:00" --org_name RoboCOIN --max_workers 16
```

- 输出目录结构
```
/home/kemove/Analyzer/stats/
└── YYYYMMDD_HHMMSS/
    ├── huggingface_dataset_downloads.csv
    ├── huggingface_top_datasets_plot.png
    ├── modelscope_dataset_downloads.csv
    └── modelscope_top_datasets_plot.png
```

## hf_analyzer_local.py（本地库版）

- 以库形式调用 HuggingFace 分析器并生成 CSV/图表
```python
from hf_analyzer_local import HuggingFaceDatasetAnalyzer
analyzer = HuggingFaceDatasetAnalyzer(org_name="RoboCOIN", save_root="stats", max_workers=8)
analyzer.run()
```

## ms_analyzer_local.py（ModelScope 脚本）

- 抓取 `RoboCOIN` 在 ModelScope 的各数据集下载量，总和并保存 CSV
```bash
python /home/kemove/Analyzer/ms_analyzer_local.py
```
输出：`robocoin_datasets_downloads.csv`

## 常见问题

- 图标仍显示默认地球：清缓存或隐私模式访问；或使用 `?favicon=/home/kemove/Analyzer/assert/logo.png`
- Matplotlib 后端警告：使用非 GUI 后端运行 `MPLBACKEND=Agg ...` 或在代码中设置 `matplotlib.use("Agg")`
- inotify ENOSPC：提高 `fs.inotify.max_user_watches` 与 `max_user_instances` 或关闭占用大量文件监控的进程

## 许可

仅用于数据统计与可视化示例。根据实际需求调整组织名称、保存目录与任务参数。