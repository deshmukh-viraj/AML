import os
import polars as pl
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional, Any, List

load_dotenv()

from src.logger import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def load_params(params_path: str='params.yaml') -> Dict[str, Any]:
    """load params from the yaml file, return empty dict on failure"""

    try:
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f) or {}
            logging.info(f"Params loaded from : {params_path}")
            return params
        logging.info(f'No params.yaml found at {params_path}')
        return {}
    
    except Exception as e:
        logging.info(f"Failded to laod params from {params_path}: {e}")
        return {}


def load_raw_data(params: dict) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    load raw parquet files saved by data_ingestion stage.
    """
    raw_dir = Path(params['storage']['raw_data_dir'])
    trans_path = raw_dir / 'transactions_raw.parquet'
    acc_path = raw_dir / 'accounts_raw.parquet'

    for p in [trans_path, acc_path]:
        if not p.exists():
            raise FileNotFoundError(f" Raw data not found at {p}")

    
    transactions = pl.scan_parquet(trans_path)
    accounts = pl.scan_parquet(acc_path)

    logger.info(f"Scanning Transactions from {trans_path} (lazy)")
    logger.info(f"Scanning Accounts from {acc_path} (lazy)")

    return transactions, accounts


#cleaning
def clean_cols_names(df: pl.LazyFrame, name: str) -> pl.LazyFrame:
    """strip trailing/leading whitespaces from the columns names"""
    cleaned = {col: col.strip() for col in df.collect_schema().names()}
    df = df.rename(cleaned)
    changed = {k: v for k, v in cleaned.items() if k != v}
    if changed:
        logger.warning(f"{name}: stripped whitespace from col names: {changed}")
    else:
        logger.info(f"{name}: col names are clean")
    return df


def parse_timestamp(df: pl.LazyFrame, timestamp_col: str) -> pl.LazyFrame:
    
    if timestamp_col not in df.collect_schema().names():
        logger.warning(f"Timestamp column '{timestamp_col}' not found, skipping parsing")
        return df
    
    schema = df.collect_schema()
    dtype = schema[timestamp_col]
    if isinstance(dtype, (pl.Date, pl.Datetime)):
        logger.info(f"Timestamp colums '{timestamp_col}' already parsed: {dtype}")
        return df

    df = df.with_columns(
        pl.col(timestamp_col).str.to_datetime(strict=False).alias(timestamp_col)
    )
    logger.info(f"Parsed '{timestamp_col}' to Datetime")
    return df


def remove_duplicates(df: pl.LazyFrame, name: str) -> pl.LazyFrame:
    
    df = df.unique(maintain_order=False)
    logger.info(f"{name}: unique() applied (lazy)")
    return df


def cast_dtypes(df: pl.LazyFrame, params: dict) -> pl.LazyFrame:
    """
    cast cols to correct dtypes as defined in params.yaml
    """
    dtypes_map = params.get('data_preprocessing', {}).get('dtype_map', {})
    if not dtypes_map:
        return df

    polars_type_map = {
        'str': pl.Utf8,
        'string': pl.Utf8,
        'float32': pl.Float32,
        'float64': pl.Float64,
        'int32': pl.Int32,
        'int64': pl.Int64,
        'bool': pl.Boolean,
        'date': pl.Date,
        'datetime': pl.Datetime
    }

    casts = []
    for col, dtype_str in dtypes_map.items():
        if col not in df.columns:
            logger.warning(f"Dtype_map column {col} not in dataframe, skipping")
            continue
        pl_type = polars_type_map.get(dtype_str.lower())
        if pl_type is None:
            logger.warning(f"Unknown dtype {dtype_str} for column {col}, skipping")
            continue

        casts.append(pl.col(col).cast(pl_type).alias(col))
        logger.info(f"Cast {col} -> {dtype_str}")

    if casts:
        df = df.with_columns(casts)
    return df


def preprocess_transactions(df: pl.LazyFrame, params: dict) -> pl.LazyFrame:
    """transactions preprocessing"""

    timestamp_col = params['data_ingestion'].get('timestamp_col', 'Timestamp')
    logger.info("----Preprocessing Transactions----")
    df = clean_cols_names(df, 'transactions')
    df = parse_timestamp(df, timestamp_col)
    df = remove_duplicates(df, 'transactions')
    df = cast_dtypes(df, params)
    return df


def preprocess_accounts(df: pl.LazyFrame, params:dict) -> pl.LazyFrame:
    """accounts preprocessing"""

    logger.info("----Preprocessing Accounts----")
    df = clean_cols_names(df, 'accounts')
    df = remove_duplicates(df, 'accounts')
    df = cast_dtypes(df, params)
    return df


def save_preprocessed_data(transacctions: pl.LazyFrame, accounts: pl.LazyFrame, params:dict) -> tuple[Path, Path]:
    """saved cleaned parquet files for feature engineering stage"""

    processed_dir = Path(params['storage']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)

    trans_path = processed_dir / 'transactions_processed.parquet'
    acc_path = processed_dir / 'accounts_processed.parquet'

    logger.info("streaming transactions to disk...")
    transacctions.sink_parquet(trans_path, compression='snappy')
    logger.info(f"Transactions -> {trans_path}")

    logger.info("streaming accounts to disk...")
    accounts.sink_parquet(acc_path, compression='snappy')
    logger.info(f"Accounts -> {acc_path}")
 

    return trans_path, acc_path


def main(params_path: str = 'params.yaml'):
    logger.info('='*60)
    logger.info('Stage 2: Data Preprocessing')
    logger.info('='*60)

    try:
        params = load_params(params_path)
        transactions, accounts = load_raw_data(params)

        transactions = preprocess_transactions(transactions, params)
        accounts = preprocess_accounts(accounts, params)

        import gc
        gc.collect()

        trans_path, acc_path = save_preprocessed_data(transactions, accounts, params)

        logger.info('='*60)
        logger.info('Data Preprocessing Complete')
        logger.info(f"Transactions -> {trans_path}")
        logger.info(f"Accounts -> {acc_path}")
        logger.info('='*60)

    except Exception as e:
        logger.error(f'Data preprocessing failed: {e}')
        raise


if __name__ == '__main__':
    main()
