
from datetime import datetime, timedelta
from pathlib import Path
import shutil
from huggingface_hub import HfApi
import pandas as pd
from prefect import task, flow
from prefect.cache_policies import INPUTS
import tushare as ts
from prefect.blocks.system import Secret

tushare_secret_block = Secret.load("tushare-token")
ts.set_token(tushare_secret_block.get())

LOCAL_FILE_PATH = "data/delta_non_fq_daily_kline.csv"
LAST_TRADE_DATE_FILE = ".last_update_date"

hf_token = Secret.load("hf-token").get()

@task
def setup_data_dir() -> None:
    """Task 1: 设置数据目录"""
    parent_folder = Path(LOCAL_FILE_PATH).parent
    if parent_folder.exists():
        shutil.rmtree(parent_folder, ignore_errors=True)
    parent_folder.mkdir(parents=True, exist_ok=True)


@task
def read_last_update_date(huggingface_repo_name: str) -> str:
    """Task 2: 读取 huggingface 上的最后一次更新日期"""
    hf_api = HfApi(token=hf_token)

    file_content = hf_api.hf_hub_download(
        repo_id=huggingface_repo_name,
        filename=LAST_TRADE_DATE_FILE,
        repo_type="dataset"
    )
    with open(file_content, "r") as f:
        last_trade_date = f.read().strip()
    return last_trade_date

@task(
    retries=3,
    cache_policy=INPUTS,
    cache_expiration=timedelta(hours=1)
)
def fetch_stock_list() -> pd.DataFrame:
    """Task 3: 获取最新的主板股票清单"""
    df = ts.pro_api().stock_basic(
        exchange='SSE,SZSE',
        fields=[
            'ts_code',
            'name',
            'area',
            'industry',
            'market',
            'exchange',
            'list_date'
        ])
    return df[df['market'] == '主板'].set_index('ts_code')

@task
def fetch_delta_data(last_update_date: str, stock_list: pd.DataFrame, end_date: str) -> str:
    """Task 4: 获取 delta 非复权日线数据"""
    last_update_date = datetime.strptime(last_update_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    
    delta_days = (end_date - last_update_date).days
    print(f"delta_days: {delta_days}")
    for i in range(1, delta_days+1):
        target_date = last_update_date + timedelta(days=i)
        target_date_str = target_date.strftime("%Y%m%d")
        print(f"正在获取 {target_date_str} 的数据")
        
        kline_df = fetch_daily_kline.submit(trade_date=target_date_str)
        limit_df = fetch_limit_price.submit(trade_date=target_date_str)
        index_df = fetch_daily_index.submit(trade_date=target_date_str)
        
        kline_df, limit_df, index_df = kline_df.result(), limit_df.result(), index_df.result()
        if kline_df.empty or limit_df.empty or index_df.empty:
            continue
        merged_df = pd.concat([kline_df, limit_df, index_df], axis=1)
        merged_df = merged_df.join(stock_list, on='ts_code', how='inner')
        merged_df.dropna(subset=['close'], inplace=True)
        
        merged_df.rename(columns={
            'pre_close': 'prev_close',
            'pct_chg': 'quote_rate',
            'vol': 'volume',
            'amount': 'turnover',
            'up_limit': 'high_limit',
            'down_limit': 'low_limit',
        }, inplace=True)
        merged_df.index = pd.MultiIndex.from_arrays([
            merged_df.index.get_level_values(level=0),
            pd.to_datetime(merged_df.index.get_level_values(level=1))
        ], names=['code', 'date'])
        merged_df['list_date'] = pd.to_datetime(merged_df['list_date'])
        
        first_write = not Path(LOCAL_FILE_PATH).exists()
        merged_df.to_csv(
            LOCAL_FILE_PATH,
            mode='w' if first_write else 'a',
            header=first_write,
            index=True
        )
    return LOCAL_FILE_PATH
    

@task(retries=3)
def fetch_daily_kline(trade_date: str) -> pd.DataFrame:
    df = ts.pro_api().daily(trade_date=trade_date)
    df = df.reset_index()
    df = df.set_index(['ts_code', 'trade_date'])
    return df[['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'amount']]


@task(retries=3)
def fetch_limit_price(trade_date: str) -> pd.DataFrame:
    df = ts.pro_api().stk_limit(trade_date=trade_date)
    df = df.reset_index()
    df = df.set_index(['ts_code', 'trade_date'])
    return df[['up_limit', 'down_limit']]


@task(retries=3)
def fetch_daily_index(trade_date: str) -> pd.DataFrame:
    df = ts.pro_api().daily_basic(trade_date=trade_date)
    df = df.reset_index()
    df = df.set_index(['ts_code', 'trade_date'])
    return df[['turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']]


@task
def upload_to_hf_datasets(file_path: str, end_trade_date: str, huggingface_repo_name: str, repo_file_name: str) -> None:
    """Task 5: 追加新记录到 huggingface datasets"""
    new_data_df = pd.read_csv(file_path, index_col=['code', 'date'])
    if new_data_df.empty:
        print(f"没有新的数据，跳过上传")
        return
    
    hf_api = HfApi(token=hf_token)
    existing_file_content = hf_api.hf_hub_download(
        repo_id=huggingface_repo_name,
        filename=repo_file_name,
        repo_type="dataset"
    )
    existing_df = pd.read_csv(existing_file_content, index_col=['code', 'date'])
    combined_df = pd.concat([existing_df, new_data_df])
    combined_df.to_csv(file_path, index=True, mode='w', header=True)
    hf_api.upload_file(
        repo_id=huggingface_repo_name,
        repo_type="dataset",
        path_or_fileobj=file_path,
        path_in_repo=repo_file_name
    )
    
    with open(LAST_TRADE_DATE_FILE, "w") as f:
        f.write(end_trade_date)
      
    hf_api.upload_file(
        repo_id=huggingface_repo_name,
        repo_type="dataset",
        path_or_fileobj=LAST_TRADE_DATE_FILE,
        path_in_repo=LAST_TRADE_DATE_FILE
    )


@flow(log_prints=True)
def fetch_delta_non_fq_daily_kline(huggingface_repo_name: str, repo_file_name: str) -> None:
    setup_data_dir()
    last_update_date = read_last_update_date(huggingface_repo_name)
    print(f"last_update_date: {last_update_date}")
    
    stock_list = fetch_stock_list()
    print(f"主板股票数量: {stock_list.shape[0]}")

    end_trade_date = datetime.now().strftime("%Y%m%d")
    local_file_path = fetch_delta_data(last_update_date, stock_list, end_trade_date)
    upload_to_hf_datasets(local_file_path, end_trade_date, huggingface_repo_name, repo_file_name)

if __name__ == "__main__":
    fetch_delta_non_fq_daily_kline(huggingface_repo_name="ellendan/a-share-prices", 
        repo_file_name="all-prices.csv")
