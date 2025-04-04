from pathlib import Path
import pandas as pd
import tushare as ts
from prefect import flow, task
from prefect.blocks.system import Secret
from huggingface_hub import HfApi
from datetime import datetime
import shutil

tushare_secret_block = Secret.load("tushare-token")
ts.set_token(tushare_secret_block.get())

HF_DATASET_NAME = "ellendan/a-share-prices"
LOCAL_FILE_PATH = "data/trade_calendar.csv"

@task
def setup_data_dir() -> None:
    """Task 1: 设置数据目录"""
    parent_folder = Path(LOCAL_FILE_PATH).parent
    if parent_folder.exists():
        shutil.rmtree(parent_folder, ignore_errors=True)
    parent_folder.mkdir(parents=True, exist_ok=True)

@task
def fetch_trade_calendar(start_date: str, end_date: str, temp_file_path: str) -> None:
    """Task 2: 获取交易日历"""
    df = ts.pro_api().trade_cal(start_date=start_date, end_date=end_date, is_open='1')
    df['cal_date'] = pd.to_datetime(df['cal_date'])
    df.sort_values('cal_date')['cal_date'].to_csv(temp_file_path, index=False)

@task
def upload_to_hf_datasets(file_path: str) -> None:
    """Task 3: 上传到 Hugging Face Datasets"""
    print("上传到 Hugging Face Datasets")
    hf_api = HfApi(token=Secret.load("hf-token").get())
    remote_csv_file_name = f"calendar.csv"
    
    hf_api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=remote_csv_file_name,
        repo_type="dataset",
        repo_id=HF_DATASET_NAME
    )
    
@flow
def fetch_trade_calendar_flow(start_date: str, end_date: str) -> None:
    """Flow: 获取交易日历"""
    setup_data_dir()
    fetch_trade_calendar(start_date, end_date, LOCAL_FILE_PATH)
    upload_to_hf_datasets(LOCAL_FILE_PATH)

if __name__ == "__main__":
    current_year = datetime.now().year
    end_date = f"{current_year}1231"
    fetch_trade_calendar_flow(start_date="20050101", end_date=end_date)
