# HuggingFace / ModelScope 数据集分析器（按文件顺序）

本项目包含三个分析器文件，按使用频率与功能排序：`hf_analyzer.py` → `hf_analyzer_local.py` → `ms_analyzer_local.py`。支持数据抓取、统计、CSV导出、可视化图表以及 Web/API 服务。

## 依赖安装

```bash
pip install huggingface_hub pandas seaborn matplotlib flask requests schedule
```

---

## hf_analyzer.py（Web/API + 定时/CLI）

- 功能：
  - 抓取 HuggingFace 组织数据集下载量，生成 CSV 与可视化 PNG
  - 提供 Web 页面和 API 接口
  - 支持 Favicon 自定义、ModelScope 对比、静态文件下载
  - 未设置 `HF_ANALYZER_FLASK=1` 时，可作为定时任务或一次性运行入口

- 启动 Web 服务：
```bash
HF_ANALYZER_FLASK=1 PORT=8000 python /home/kemove/Analyzer/hf_analyzer.py
```
  - 访问 `http://localhost:8000/` 查看可视化页面
  - 可通过 URL 参数控制：`ms_org_name`、`save_root`、`max_workers`、`favicon`

- 常用接口：
  - `GET /health`：健康检查
  - `GET /`：渲染页面并执行分析，输出 CSV 与图表
  - `POST /analyze`：触发分析，返回统计摘要（JSON）
  - `GET /static_plot?path=<abs_path>`：返回图表 PNG
  - `GET /static_csv?path=<abs_path>`：下载 CSV
  - `GET /static_favicon?path=<abs_path>`：按扩展名返回 PNG/ICO/JPEG 图标
  - `GET /favicon.ico`：默认站点图标（支持回退到 PNG）

- API 示例：
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
        "org_name": "RoboCOIN",
        "ms_org_name": "RoboCOIN",
        "save_root": "/home/kemove/Analyzer/stats",
        "max_workers": 8
      }'
```

- 自定义图标：
  - 环境变量：`FAVICON_PATH=/home/kemove/Analyzer/assert/logo.ico`
  - URL 参数：`http://localhost:8000/?favicon=/home/kemove/Analyzer/assert/logo.png`

- 定时/CLI 运行：
```bash
# 立即执行一次（直接运行任务）
python /home/kemove/Analyzer/hf_analyzer.py --now

# 每天 00:00 执行（需安装 schedule 包）
python /home/kemove/Analyzer/hf_analyzer.py --time "00:00" --org_name "RoboCOIN" --max_workers 8
```

- 输出目录：
  - 使用 `save_root/YYYYMMDD_HHMMSS/` 结构，包含：
    - `dataset_downloads.csv`
    - `analysis_summary.csv`
    - `top_datasets_plot.png`

---

## hf_analyzer_local.py（本地库版 HF 分析器）

- 用途：在本地以库方式运行 HuggingFace 数据分析与图表生成（保存到 `stats/`）。
- 典型使用（Python 交互/脚本中）：
```python
from hf_analyzer_local import HuggingFaceDatasetAnalyzer

analyzer = HuggingFaceDatasetAnalyzer(org_name="RoboCOIN", save_root="stats", max_workers=8)
analyzer.run()  # 将生成 CSV 与 top_datasets_plot.png
```

- 说明：该文件不包含 CLI 入口（无 `if __name__ == "__main__"`），推荐按上述方式导入调用。

---

## ms_analyzer_local.py（ModelScope 下载统计脚本）

- 用途：抓取 `RoboCOIN` 在 ModelScope 的各数据集下载量并汇总总和，保存 CSV。
- 运行：
```bash
python /home/kemove/Analyzer/ms_analyzer_local.py
```
- 输出：
  - 控制台打印总下载量与数据集列表
  - 生成 `robocoin_datasets_downloads.csv`

---

## 许可

仅用于数据统计与可视化示例。根据实际需求调整组织名称、保存目录与任务参数。