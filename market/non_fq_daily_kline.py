from pathlib import Path
from datetime import datetime, timedelta
from huggingface_hub import HfApi

from pandas import DataFrame, Series
import tushare as ts
import pandas as pd

from prefect import flow, task
from prefect.blocks.system import Secret
from prefect.cache_policies import INPUTS
from prefect.concurrency.sync import rate_limit

LOCAL_FILE_PATH = "non_fq_daily_klines.csv"
HF_DATASET_NAME = "ellendan/a-share-prices"

tushare_secret_block = Secret.load("tushare-token")
ts.set_token(tushare_secret_block.get())


@task(
    retries=3,
    cache_policy=INPUTS,
    cache_expiration=timedelta(hours=1)
)
def fetch_stock_list() -> DataFrame:
    """Task 1: 获取最新的主板股票清单"""
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
    return df[df['market'] == '主板']


@task(retries=3)
def fetch_daily_kline(ts_code: str, start_date: str, end_date: str) -> DataFrame:
    df = ts.pro_api().daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.reset_index()
    df = df.set_index(['ts_code', 'trade_date'])
    return df[['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'amount']]


@task(retries=3)
def fetch_limit_price(ts_code: str, start_date: str, end_date: str) -> DataFrame:
    df = ts.pro_api().stk_limit(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.reset_index()
    df = df.set_index(['ts_code', 'trade_date'])
    return df[['up_limit', 'down_limit']]


@task(retries=3)
def fetch_daily_index(ts_code: str, start_date: str, end_date: str) -> DataFrame:
    df = ts.pro_api().daily_basic(ts_code=ts_code,
                                  start_date=start_date, end_date=end_date)
    df = df.reset_index()
    df = df.set_index(['ts_code', 'trade_date'])
    return df[['turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']]


@task
def fetch_daily_price(stock_series: Series, start_date: str, end_date: str) -> DataFrame:
    """Task 2: 抓取非复权日线数据"""
    rate_limit("tushare-daily-api")
    ts_code = stock_series['ts_code']
    kline_df = fetch_daily_kline(
        ts_code=ts_code, start_date=start_date, end_date=end_date)
    limit_df = fetch_limit_price(
        ts_code=ts_code, start_date=start_date, end_date=end_date)
    index_df = fetch_daily_index(
        ts_code=ts_code, start_date=start_date, end_date=end_date)
    merged_df = pd.concat([kline_df, limit_df, index_df], axis=1)

    stock_info_df = stock_series.to_frame().transpose()
    stock_info_df['list_date'] = pd.to_datetime(stock_info_df['list_date'])
    stock_info_df.set_index('ts_code', inplace=True)
    merged_df = merged_df.join(stock_info_df, on='ts_code')
    
    merged_df.dropna(subset=['close'], inplace=True)
    return merged_df

@task
def append_to_csv(stock_df: DataFrame, file_path: str) -> None:
    """Task 3: 写入 csv"""
    stock_df.rename(columns={
        'pre_close': 'prev_close',
        'pct_chg': 'quote_rate',
        'vol': 'volume',
        'amount': 'turnover',
        'up_limit': 'high_limit',
        'down_limit': 'low_limit',
    }, inplace=True)
    stock_df.index = pd.MultiIndex.from_arrays([
        stock_df.index.get_level_values(level=0),
        pd.to_datetime(stock_df.index.get_level_values(level=1))
    ], names=['code', 'date'])
    
    first_write = not Path(file_path).exists()
    stock_df.to_csv(
        file_path,
        mode='w' if first_write else 'a',
        header=first_write,
        index=True
    )

@task
def upload_to_hf_datasets(file_path: str, end_date: str) -> None:
    """Task 4: 上传到 Hugging Face Datasets"""
    print("上传到 Hugging Face Datasets")
    hf_api = HfApi(token=Secret.load("hf-token").get())
    csv_file_name = f"non-fq-daily-klines-{end_date}.csv"
    
    hf_api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=csv_file_name,
        repo_type="dataset",
        repo_id=HF_DATASET_NAME
    )

@flow(log_prints=True)
def fetch_non_fq_daily_kline() -> None:
    """Flow: 抓取非复权日线数据"""
    stock_list = fetch_stock_list()
    print(f"主板股票数量: {stock_list.shape[0]}")

    end_date = datetime.now().strftime("%Y%m%d")
    for _, row in stock_list.iterrows():
        print(f"正在获取 {row['ts_code']} 的日线数据")
        kline_df = fetch_daily_price(row, '20050101', end_date)
        append_to_csv(kline_df, LOCAL_FILE_PATH)

    upload_to_hf_datasets(LOCAL_FILE_PATH, end_date)


# Run the flow
if __name__ == "__main__":
    fetch_non_fq_daily_kline()
