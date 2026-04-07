import gc
import sys
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
import polars as pl

load_dotenv()
try:
    from src.logger import logging
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

try:
    from src.features.build_features import build_all_features
    from src.features.experimental.toxic_corridors import derive_toxic_corridors
except ImportError as e:
    logger.error(f"AML feature module not found: {e}")


#load params
def load_params(params_path: str = 'params.yaml') -> dict:
    """load pipeline parameters from thr params.yaml"""

    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Parameters loaded from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"params.yaml not found at {params_path}")
        raise


#load preprocessed data
def load_clean_data(params:dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    load clean parquet files from preprocessing stage
    transactions loaded as lazyframe - build_features.py
    expects lazyframe input for memory efficient batch processing
    accountd loaded as eagar dataframe
    """
    processed_dir = Path(params['storage']['processed_dir'])
    trans_path = processed_dir / "transactions_processed.parquet"
    acc_path = processed_dir / "accounts_processed.parquet"

    for p in [trans_path, acc_path]:
        if not p.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {p}.")

    #lazyframe
    transactions = pl.scan_parquet(trans_path)
    accounts = pl.scan_parquet(acc_path)
    n_rows = transactions.select(pl.len()).collect().item()
    logger.info(f"Transactions (lazy): {n_rows:,} ros")
    logger.info(f"Accounts (eager): {len(accounts):,} rows")

    return transactions, accounts


#feature engineering
def run_feature_engg(transactions: pl.LazyFrame, accounts: pl.DataFrame, params: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    """
    orchestrates the full AML feature engineering pipeline.
    calls build_all_features() from the validated exploration phase
    """
    logger.info('Starting AML feature engineering pipeline...')
    output_dir.mkdir(parents=True, exit_ok=True)

    #build_alll_features 
    train_path, val_path, test_path = build_all_features(
        transactions_path=None, 
        accounts_path=None,
        output_dir=output_dir,
        trans_lazy=transactions,
        acc_df = accounts,
    )
    logger.info(f"Train features -> {train_path}")
    logger.info(f"Val feature -> {val_path}")
    logger.info(f"Test feature -> {test_path}")

    return train_path, val_path, test_path


#main
def main(params_path: str='params.yaml'):
    logger.info('='*60)
    logger.info('Stage 3: Feature Engineering')
    logger.info('='*60)

    try:
        params = load_params(params_path)
        output_dir = Path(params['storage']['features_dir'])

        #load clean data from stage 2
        transactions, accounts = load_clean_data(params)
        #run feature engineeing
        train_path, val_path, test_path = run_feature_engg(
            transactions, accounts, params, output_dir
        )
        gc.collect()

        logger.info('='*60)
        logger.info('Featune engineering complete')
        logger.info(f"Train -> {train_path}")
        logger.info(f"Val -> {val_path}")
        logger.info(f"Test -> {test_path}")
        logger.info('='*60)
    
    except Exception as e:
        logger.error(f'Feature engineering failed: {e}')
        raise


if __name__ == '__main__':
    main()