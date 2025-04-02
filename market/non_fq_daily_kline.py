from typing import Any, Optional
from datetime import timedelta

from pandas import DataFrame
import tushare as ts

from prefect import flow, task
from prefect.blocks.system import Secret
from prefect.cache_policies import INPUTS
from prefect.concurrency.sync import rate_limit

secret_block = Secret.load("tushare-token")
ts.set_token(secret_block.get())
        
@task(
    retries=3,
    cache_policy=INPUTS,
    cache_expiration=timedelta(hours=1)
)
def fetch_stock_list() -> DataFrame:
    """Task 1: 获取最新的主板股票清单"""
    rate_limit("tushare-daily-api")
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

@flow(log_prints=True)
def fetch_non_fq_daily_kline() -> None:
    """Flow: 抓取非复权日线数据"""
    stock_list = fetch_stock_list()
    print(f"主板股票数量: {stock_list.shape[0]}")


# Run the flow
if __name__ == "__main__":
    fetch_non_fq_daily_kline()