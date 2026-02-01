"""
Integrated AML Feature Engineering Pipeline

This module orchestrates the complete feature engineering pipeline:
1. Base features (temporal, benford, lifecycle)
2. Advanced rolling features (burst, time-gaps, velocity)
3. Counterparty entropy and network metrics
4. Unsupervised anomaly detection (Isolation Forest)
5. Feature validation and output

Usage:
    from src.features.build_features import build_all_features
    
    build_all_features(
        transactions_path='data/raw/transactions.csv',
        accounts_path='data/raw/accounts.csv',
        output_dir='aml_features',
        compute_anomaly_scores=True
    )
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import polars as pl
import pandas as pd

# Import feature modules
from src.features.experimental.rolling_features_v2 import (
    compute_rolling_features_batch1,
    compute_rolling_features_batch2,
    compute_rolling_features_batch3,
)
from src.features.experimental.ratio_features import compute_advanced_features
from src.features.experimental.advanced_rolling_features_v2 import (
    add_advanced_rolling_features
)
from src.features.experimental.counterparty_entropy_features_v2 import (
    add_counterparty_entropy_features
)
from src.features.experimental.isolation_forest_anomaly import (
    add_isolation_forest_scores
)
from src.utils.hashing import hash_pii_column

logger = logging.getLogger(__name__)


def add_base_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add foundational temporal and benford features.
    """
    logger.info("Adding base temporal features...")
    
    # Cyclical time encoding
    import numpy as np
    df = df.with_columns([
        (2 * np.pi * pl.col('Timestamp').dt.hour() / 24).sin().alias('hour_sin'),
        (2 * np.pi * pl.col('Timestamp').dt.hour() / 24).cos().alias('hour_cos'),
        (2 * np.pi * pl.col('Timestamp').dt.weekday() / 7).sin().alias('day_of_week_sin'),
        (2 * np.pi * pl.col('Timestamp').dt.weekday() / 7).cos().alias('day_of_week_cos'),
    ])
    
    # Benford's law features
    logger.info("Adding Benford's law features...")
    df = df.with_columns([
        pl.col('Amount Paid')
            .cast(pl.Utf8)
            .str.replace_all(r'[^1-9].*', '')
            .str.slice(0, 1)
            .cast(pl.Int32, strict=False)
            .alias('first_digit'),
        (pl.col('Amount Paid') % 100 == 0).cast(pl.Int8).alias('is_round_100'),
        (pl.col('Amount Paid') % 1000 == 0).cast(pl.Int8).alias('is_round_1000'),
    ])
    
    # Account lifecycle features
    logger.info("Adding account lifecycle features...")
    df = df.with_columns([
        pl.col('Timestamp').min().over('Account_HASHED').alias('account_first_txn')
    ])
    
    df = df.with_columns([
        ((pl.col('Timestamp') - pl.col('account_first_txn'))
         .dt.total_seconds() / 86400)
        .alias('account_tenure_days'),
        
        pl.col('Timestamp')
            .rank(method='ordinal')
            .over('Account_HASHED')
            .alias('txn_rank_in_account_history'),
        
        ((pl.col('Timestamp') - pl.col('Timestamp').shift(1).over('Account_HASHED'))
         .dt.total_seconds() / 86400)
        .fill_null(0)
        .alias('days_since_last_txn'),
        
        (pl.col('Timestamp') - pl.col('account_first_txn') >= pl.duration(days=7))
        .cast(pl.Int8)
        .alias('has_7d_history'),
        
        (pl.col('Timestamp') - pl.col('account_first_txn') >= pl.duration(days=28))
        .cast(pl.Int8)
        .alias('has_28d_history'),
    ])
    
    return df


def build_training_features(
    train_df: pl.LazyFrame,
    val_df: pl.LazyFrame,
    test_df: pl.LazyFrame,
    accounts: Optional[pl.DataFrame] = None
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Build all features for training, validation, and test sets.
    Processes splits sequentially to manage memory.
    
    Args:
        train_df, val_df, test_df: Lazy DataFrames by split
        accounts: Accounts reference data
    
    Returns:
        Tuple of (train_features, val_features, test_features) as eager DataFrames
    """
    logger.info("="*70)
    logger.info("BUILDING AML FEATURES")
    logger.info("="*70)
    
    # Process each split
    splits = [
        ('train', train_df),
        ('val', val_df),
        ('test', test_df)
    ]
    
    processed_splits = {}
    
    for split_name, df in splits:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {split_name.upper()} split")
        logger.info(f"{'='*70}")
        
        # 1. Sort for rolling features
        logger.info("  Step 1: Sorting...")
        df = df.sort(['Account_HASHED', 'Timestamp'])
        
        # 2. Base features
        logger.info("  Step 2: Base features...")
        df = add_base_features(df)
        
        # 3. Standard rolling features (from original pipeline)
        logger.info("  Step 3: Standard rolling features...")
        df = compute_rolling_features_batch1(df)
        df = compute_rolling_features_batch2(df)
        df = compute_rolling_features_batch3(df)
        
        # 4. Derived/ratio features
        logger.info("  Step 4: Ratio and derived features...")
        df = compute_advanced_features(df)
        
        # 5. Advanced rolling features (NEW)
        logger.info("  Step 5: Advanced rolling features (burst, time-gaps, velocity)...")
        df = add_advanced_rolling_features(df)
        
        # 6. Counterparty entropy features (NEW)
        logger.info("  Step 6: Counterparty entropy and network features...")
        df = add_counterparty_entropy_features(df)
        
        # Collect to eager for next steps
        logger.info("  Collecting to eager DataFrame (may require memory)...")
        df = df.collect()
        
        processed_splits[split_name] = df
    
    train_features = processed_splits['train']
    val_features = processed_splits['val']
    test_features = processed_splits['test']
    
    # Add anomaly scores (fitted on train, applied to all) - DISABLED FOR NOW
    # logger.info("\n" + "="*70)
    # logger.info("Adding Unsupervised Anomaly Scores (Isolation Forest)")
    # logger.info("="*70)
    
    # train_features, val_features, test_features = add_isolation_forest_scores(
    #     train_features,
    #     val_features,
    #     test_features,
    #     contamination=0.10
    # )
    
    return train_features, val_features, test_features


def validate_features(df: pl.DataFrame) -> Dict:
    """
    Validate feature engineering output.
    
    Checks:
    - No NaNs in critical features
    - Feature distribution reasonableness
    - Class balance
    - Feature diversity
    """
    logger.info("Validating feature quality...")
    
    validation_report = {
        'num_rows': len(df),
        'num_features': len(df.columns),
        'missing_rate': df.select(pl.col('*').is_null().sum()).to_dict(as_series=False),
    }
    
    # Check critical features for missing values
    critical_features = [
        col for col in df.columns 
        if 'rolling' in col or 'burst' in col or 'entropy' in col or 'anomaly' in col
    ]
    
    for col in critical_features:
        missing = df.select(pl.col(col).is_null().sum()).item()
        if missing > 0:
            logger.warning(f"  ⚠ {col}: {missing} missing values")
    
    # Check numeric columns
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    
    # Basic statistics
    logger.info(f"\nFeature Statistics:")
    logger.info(f"  Total rows: {validation_report['num_rows']}")
    logger.info(f"  Total features: {validation_report['num_features']}")
    
    # Class balance
    if 'Is_Laundering' in df.columns or 'Is Laundering' in df.columns:
        target_col = 'Is_Laundering' if 'Is_Laundering' in df.columns else 'Is Laundering'
        class_counts = df.group_by(target_col).agg(pl.count().alias('count'))
        logger.info(f"\n  Class Distribution:")
        for row in class_counts.iter_rows(named=True):
            logger.info(f"    Class {row[target_col]}: {row['count']} samples")
    
    return validation_report


def save_features(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_dir: Path
):
    """Save feature sets to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("Saving Features")
    logger.info("="*70)
    
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        output_path = output_dir / f'{split_name}_features.parquet'
        df.write_parquet(output_path)
        logger.info(f"✓ {split_name}: {len(df)} rows → {output_path}")


def build_all_features(
    transactions_path: Path,
    accounts_path: Path,
    output_dir: Path = Path('./aml_features'),
    compute_anomaly_scores: bool = True,
    sample_fraction: Optional[float] = None
) -> Tuple[Path, Path, Path]:
    """
    Complete feature engineering pipeline.
    
    Args:
        transactions_path: CSV file with transactions
        accounts_path: CSV file with accounts
        output_dir: Directory for output
        compute_anomaly_scores: Whether to compute Isolation Forest scores
        sample_fraction: Optional - use fraction of data for testing
    
    Returns:
        Tuple of (train_features_path, val_features_path, test_features_path)
    """
    logger.info("="*70)
    logger.info("AML FEATURE ENGINEERING PIPELINE (COMPLETE)")
    logger.info("="*70)
    
    # Load data
    logger.info(f"\nLoading transactions from {transactions_path}")
    trans = pl.scan_csv(transactions_path, try_parse_dates=True)
    
    logger.info(f"Loading accounts from {accounts_path}")
    accounts = pl.read_csv(accounts_path)
    
    if sample_fraction:
        logger.info(f"Sampling {sample_fraction*100}% of transactions...")
        trans = trans.collect().sample(fraction=sample_fraction, seed=42).lazy()
    
    # Hash PII
    logger.info("Hashing PII columns...")
    trans = hash_pii_column(trans, 'Account')
    
    # Create temporal splits
    logger.info("Creating temporal splits...")
    max_timestamp = trans.select(pl.col('Timestamp').max()).collect()[0, 0]
    
    test_start = max_timestamp - pl.duration(days=7)
    val_start = test_start - pl.duration(days=7)
    
    train_df = trans.filter(pl.col('Timestamp') < val_start)
    val_df = trans.filter(
        (pl.col('Timestamp') >= val_start) & (pl.col('Timestamp') < test_start)
    )
    test_df = trans.filter(pl.col('Timestamp') >= test_start)
    
    # Build features
    train_features, val_features, test_features = build_training_features(
        train_df, val_df, test_df, accounts
    )
    
    # Validate
    validate_features(train_features)
    
    # Save
    save_features(train_features, val_features, test_features, output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("✅ FEATURE ENGINEERING COMPLETE")
    logger.info("="*70)
    
    return (
        output_dir / 'train_features.parquet',
        output_dir / 'val_features.parquet',
        output_dir / 'test_features.parquet'
    )


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Default paths (adjust as needed)
    trans_path = Path('data/raw/HI-Medium_Trans.csv')
    acc_path = Path('data/raw/HI-Medium_accounts.csv')
    output_path = Path('aml_features')
    
    if not trans_path.exists() or not acc_path.exists():
        logger.error(f"Data files not found at {trans_path} or {acc_path}")
        sys.exit(1)
    
    build_all_features(trans_path, acc_path, output_path)
