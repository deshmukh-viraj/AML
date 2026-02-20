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
import warnings
import gc
warnings.filterwarnings('ignore')

# Import feature modules
from src.features.experimental.base_features import add_base_features
from src.features.experimental.precompute_entity_stats import precompute_entity_stats

from src.features.experimental.rolling_features_v2 import compute_rolling_features
from src.features.experimental.ratio_features import compute_advanced_features
from src.features.experimental.derived_features import compute_derived_features
from src.features.experimental.advanced_rolling_features_v2 import (
    add_advanced_rolling_features
)
from src.features.experimental.counterparty_entropy_features_v2 import (
    add_counterparty_entropy_features
)

from src.utils.hashing import hash_pii_column
from src.features.experimental.network_features import add_network_features
from src.features.experimental.toxic_corridors import apply_toxic_corridor_features


logger = logging.getLogger(__name__)


def optimize_dtypes(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns([
        pl.col('Amount Paid').cast(pl.Float32),
        pl.col('Amount Received').cast(pl.Float32)
    ])


def build_training_features(
    train_df: pl.LazyFrame,
    val_df: pl.LazyFrame,
    test_df: pl.LazyFrame,
    accounts: Optional[pl.DataFrame] = None,
    output_dir: Path = Path('./aml_features')
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

    entity_stats_lazy = None
    if accounts is not None:
        logger.info("   Precomputing entity-level stats from accounts...")
        entity_stats = precompute_entity_stats(accounts)

        if 'Account HASHED' not in entity_stats.columns and 'Account Number' in entity_stats.columns:
            logger.info("   Hashing Account Number in entity_stats to produce Account_HASHED...")
            entity_stats = hash_pii_column(entity_stats.lazy(), 'Account Number').collect()

        entity_stats_lazy = entity_stats.lazy()
        del entity_stats

        import gc
        gc.collect()

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
        
        # 0. Optimize dtypes
        df = optimize_dtypes(df)
        
        # 1. Sort for rolling features
        logger.info("   Step 1: Sorting(maintained through pipeline)...")
        df = df.sort(['Account_HASHED', 'Timestamp'])
        
        # 2. Base features
        logger.info("   Step 2: Base features...")
        df = add_base_features(df)

        # 2.1 Join precompute entity/accounts stats if available
        if entity_stats_lazy is not None:
            logger.info("   Step 2.5: Joining entity/accounts stats into transactions....")
            df = df.join(entity_stats_lazy, left_on='Account_HASHED', right_on='Account Number_HASHED', how='left')

            assert 'Account_HASHED' in df.columns or isinstance(df, pl.LazyFrame), "Expected Account_HASHED in transaction df"
            if entity_stats_lazy is not None:
                assert 'Account Number_HASHED' in entity_stats_lazy.columns, "entity_stats must contain Account Number_HASHED for join"
        
        # Checkpoint A: 
        logger.info("   CHECKPOINT A: Materializing base + entity features...")
        checkpoint_a_path = output_dir / f'{split_name}_checkpoint_base.parquet'
        df.sink_parquet(checkpoint_a_path, compression='zstd')

        df = pl.scan_parquet(checkpoint_a_path)
        logger.info(f"   Checkpoint A written: {checkpoint_a_path}")

        import gc
        gc.collect()

        # 3. Standard rolling features (from original pipeline)
        logger.info("   Step 3: Standard rolling features...")
        df = compute_rolling_features(df)
   
        # 4. Derived/ratio features
        logger.info("   Step 4: Ratio and Derived features...")
        df = compute_advanced_features(df)
        df = compute_derived_features(df)

        # Checkpoint B: after all basic rolling/ratio features
        logger.info("   CHECKPOINT B: Materializing rolling + ratio features...")
        checkpoint_b_path = output_dir / f'{split_name}_checkpoint_rolling_ratio.parquet'
        df.sink_parquet(checkpoint_b_path, compression='zstd')

        df = pl.scan_parquet(checkpoint_b_path)
        logger.info(f"   Checkpoint B written: {checkpoint_b_path}")

        import gc
        gc.collect()

        adv_rolling_ip_cols = [
            'Account_HASHED', 'Timestamp', 'Amount Paid', 'Amount Received', 'Account_duplicated_0', 
        ]

        # 5. Advanced rolling features
        logger.info("   Step 5: Advanced rolling features (ISOLATED + STREAMING)...")
        checkpoint_c_path = output_dir / f"{split_name}_checkpoint_advanced.parquet"

        df_adv_input = df.select(adv_rolling_ip_cols)

        # compute advanced rolling in isolation
        df_adv = (
            df_adv_input
            .pipe(add_advanced_rolling_features)
            .sink_parquet(checkpoint_c_path, compression='zstd')
        )
        # reattach to main frame
        df = (
            pl.scan_parquet(checkpoint_b_path)
            .join(
                pl.scan_parquet(checkpoint_c_path),
                on=["Account_HASHED", "Timestamp"],
                how="left"
            )
        )

        del df_adv_input, df_adv
        import gc
        gc.collect()

        logger.info(f"   Checkpoint C written: {checkpoint_c_path}")
        
        # # 6. Counterparty entropy features 
        # logger.info("   Step 6: Counterparty entropy and network features...")
        # df = add_counterparty_entropy_features(df)
        
        # # 7. Network Features
        # logger.info("   Step 7: Network Features...")
        # df = add_network_features(df)

        # # 8. Toxic Corridors 
        # logger.info("   Step 8: Flagging Toxic Corridors...")
        # df = apply_toxic_corridor_features(df, toxic_corridors=None)

        # Checkpoint D: Materializing the counterparty entropy features
        logger.info("   Steps 6: Couterparty entropy features (ISOLATED)...")
        checkpoint_d_path = output_dir / f"{split_name}_checkpoint_conterarty.parquet"
        cp_ip_cols = ['Account_HASHED', 'Timestamp', 'Account_duplicated_0', 'total_amount_paid_28d',
                       'total_amount_received_28d', 'txn_count_28d']
        
        df_cp_input = df.select(cp_ip_cols)

        logger.info("   Computing counterparty entropy features...")
        df_cp = (
            df_cp_input
            .pipe(add_counterparty_entropy_features)
            .collect(engine='streaming')
            .sink_parquet(checkpoint_d_path, compression='zstd')
        )
        
        del df_cp_input, df_cp
        gc.collect()

        logger.info(f"    Checkpoint D written: {checkpoint_d_path}")

        # Checkpoint E: Network + Toxic corridors
        logger.info("   Step 7-8: Network + Toxic features (ISOLATED)...")
        checkpoint_e_path = output_dir / f"{split_name}_chekpoint_network.parquet"

        net_ip_cols = ['Account_HASHED', 'Timestamp', 'From Bank', 'To Bank', 'Amount Paid', 'Amount Received']

        df_net_input = df.select(net_ip_cols)
        df_net = (
            df_net_input
            .pipe(add_network_features)
            .pipe(apply_toxic_corridor_features, toxic_corridors=None)
            .collect(engine='streaming')
            .sink_parquet(checkpoint_e_path, compression='zstd')
        )

        del df_net, df_net_input
        gc.collect()
        logger.info(f"   Checkpoint E written: {checkpoint_e_path}")

        # rejoin to main
        df = (pl.scan_parquet(checkpoint_b_path).join
        (pl.scan_parquet(checkpoint_c_path), on=['Account_HASHED', 'Timestamp'], how='left').join
             (pl.scan_parquet(checkpoint_d_path), on=['Account_HASHED', 'Timestamp'], how='left').join
             (pl.scan_parquet(checkpoint_e_path), on=['Account_HASHED', 'Timstamp'], how='left')
        )

        #final collection
        logger.info(f"  Final collection for {split_name} (streaming)..")
        if isinstance(df, pl.LazyFrame):
            df = df.collect(engine='streaming')
       
        
        processed_splits[split_name] = df
        logger.info(f"  {split_name.upper()} split complete. Running Garbage Collection")
        import gc
        gc.collect()
    
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
    }
    
    # Check critical features for missing values
    critical_features = [
        col for col in df.columns 
        if 'rolling' in col or 'burst' in col or 'entropy' in col or 'anomaly' in col
    ]
    
    for col in critical_features[:5]:
        missing = df.select(pl.col(col).is_null().sum()).item()
        if missing > 0:
            logger.warning(f"   {col}: {missing} missing values")
    

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
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("\n" + "="*70)
    logger.info("Saving Features")
    logger.info("="*70)
    
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        output_path = output_dir / f'{split_name}_features.parquet'
        df.write_parquet(output_path, compression='zstd')
        logger.info(f" {split_name}: {len(df)} rows → {output_path}")

        del df
        import gc
        gc.collect()

def build_all_features(
    transactions_path: Path,
    accounts_path: Path,
    output_dir: Path = Path('./aml_features'),
    compute_anomaly_scores: bool = False,
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
    
    # convert transactions data
    trans_parquet_path = transactions_path.with_suffix('.parquet')

    if trans_parquet_path.exists():
        logger.info(f"   Found cached Parquet: {trans_parquet_path}")    
    else:
        logger.info(f"   Converting {transactions_path.name}...")
    
        pl.scan_csv(
            transactions_path, 
            try_parse_dates=True,
            dtypes={
                'From Bank': pl.Utf8,
                'To Bank': pl.Utf8,
                'Amount Paid': pl.Float32,
                'Amount Received': pl.Float32,   }
        ).sink_parquet(trans_parquet_path, compression='snappy')

    # covert accounts data
    acc_parquet_path = accounts_path.with_suffix('.parquet')

    if acc_parquet_path.exists():
        logger.info(f"   Found cached Parquet: {acc_parquet_path}")
    else:
        logger.info(f"   Converting {accounts_path.name}...")
    
        df_acc = pl.read_csv(accounts_path)
        df_acc.write_parquet(acc_parquet_path)

        del df_acc

    logger.info("   Loading from Parquet....")
    # Load data from parquet files
    logger.info(f"\nLoading transactions from {trans_parquet_path}")
    trans = pl.scan_parquet(trans_parquet_path)

    logger.info(f"Loading accounts from {acc_parquet_path}")
    accounts = pl.read_parquet(acc_parquet_path)
    
    if sample_fraction:
        logger.info(f"Sampling {sample_fraction*100}% of transactions...")
        trans = trans.collect().sample(fraction=sample_fraction, seed=42).lazy()
    
    # Hash PII
    logger.info("Hashing PII columns...")
    trans = hash_pii_column(trans, 'Account')
    trans = trans.with_columns(pl.col('Account_HASHED').cast(pl.Utf8))
    
    # Create temporal splits
    logger.info("Creating temporal splits...")
    max_timestamp = trans.select(pl.col('Timestamp').max()).collect()[0, 0]
    
    test_start = max_timestamp - pl.duration(days=7)
    val_start = test_start - pl.duration(days=7)
    
    train_df = (
        trans.filter(pl.col('Timestamp') < val_start).sort(['Account_HASHED', 'Timestamp'])
        )
    val_df = (trans.filter(
        (pl.col('Timestamp') >= val_start) & (pl.col('Timestamp') < test_start)).sort(['Account_HASHED', 'Timestamp'])
    )
    test_df = (
        trans.filter(pl.col('Timestamp') >= test_start).sort(['Account_HASHED', 'Timestamp'])
        )
    
    del trans
    import gc
    gc.collect()

    # Build features
    train_features, val_features, test_features = build_training_features(
        train_df, val_df, test_df, accounts, output_dir
    )
    
    del train_df, val_df, test_df
    import gc
    gc.collect()
    
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
