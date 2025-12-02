# 需求：从 https://modelscope.cn/organization/RoboCOIN?tab=dataset 上抓取每一个数据集的下载量，并计算所有数据集下载量的总和

import requests
import pandas as pd

def _get_total_downloads():
    base_url = "https://modelscope.cn/api/v1/datasets"
    params = {
        "owner": "RoboCOIN",
        "page": 1,
        "PageSize": 1000 # Assuming there are less than 1000 datasets
    }

    total_downloads = 0
    datasets_info = []
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return 0, []

    data = response.json()
    if data.get("Code") != 200:
        print(f"API Error: {data}")
        return 0, []

    datasets = data.get("Data", [])
    print(f"Found {len(datasets)} datasets")

    for dataset in datasets:
        name = f"{dataset.get('Namespace')}/{dataset.get('Name')}"
        downloads = dataset.get("Downloads", 0)
        total_downloads += downloads
        datasets_info.append({"Dataset": name, "Downloads": downloads})

    return total_downloads, datasets_info

def get_ms_downloads():  # IGNORE
    total, info = _get_total_downloads()
    return total

if __name__ == "__main__":
    total, datasets_info = _get_total_downloads()
    print(f"Total downloads: {total}")
    
    # Save to CSV
    df = pd.DataFrame(datasets_info)
    df.to_csv("robocoin_datasets_downloads.csv", index=False)
    print("Data saved to robocoin_datasets_downloads.csv")
