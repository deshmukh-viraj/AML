# -*- coding: utf-8 -*-
"""
Advanced rolling features for AML detection - MEMORY-SAFE LazyFrame version.

CRITICAL CHANGES:
1. All functions accept/return LazyFrame
2. Single sort at entry, maintained throughout
3. Rolling operations use streaming-compatible integer windows
4. No intermediate collects
"""

import polars as pl
import numpy as np
from typing import List


def compute_burst_score(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Detect transaction clustering (smurfing indicator) - LAZY VERSION.
    """
    # Already sorted by caller, no re-sort
    
    # 1. Hourly truncation and count in single expression
    df = df.with_columns([
        pl.col('Timestamp').dt.truncate('1h').alias('hour_window')
    ])
    
    # Use rolling count on integer index instead of groupby
    df = df.with_columns([
        pl.col('Timestamp').count().over(['Account_HASHED', 'hour_window']).cast(pl.UInt32).alias('txn_in_hour')
    ])
    
    # 2. Rolling mean on txn count (48-hour = ~500 rows assuming regular txns)
    df = df.with_columns([
        pl.col('txn_in_hour').cast(pl.Float32)
            .rolling_mean(window_size=48)
            .over('Account_HASHED')
            .fill_null(1.0)
            .alias('baseline_txn_per_hour_24h')
    ])
    
    # 3. Burst calculations
    df = df.with_columns([
        ((pl.col('txn_in_hour').cast(pl.Float32) / 
          (pl.col('baseline_txn_per_hour_24h') + 1.0)) - 1.0)
        .cast(pl.Float32)
        .alias('burst_score_1h'),
        
        (pl.col('txn_in_hour') > (pl.col('baseline_txn_per_hour_24h') * 2.0))
        .cast(pl.Int8)
        .rolling_sum(window_size=24)
        .over('Account_HASHED')
        .fill_null(0)
        .cast(pl.UInt32)
        .alias('burst_count_24h')
    ])
    
    return df.drop('hour_window')


def compute_timegap_statistics(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute inter-transaction time statistics - LAZY VERSION.
    """
    # Time gap to previous
    df = df.with_columns([
        ((pl.col('Timestamp') - pl.col('Timestamp').shift(1).over('Account_HASHED'))
         .dt.total_seconds() / 60.0)
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias('minutes_since_last_txn')
    ])
    
    # Rolling stats (500-row window)
    df = df.with_columns([
        pl.col('minutes_since_last_txn')
            .rolling_mean(window_size=500)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('avg_timegap_minutes_28d'),
            
        pl.col('minutes_since_last_txn')
            .rolling_min(window_size=500)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('min_timegap_minutes_28d'),
            
        pl.col('minutes_since_last_txn')
            .rolling_max(window_size=500)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('max_timegap_minutes_28d'),
            
        pl.col('minutes_since_last_txn')
            .rolling_std(window_size=500)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('std_timegap_minutes_28d'),
    ])
    
    # Derived metrics
    df = df.with_columns([
        (pl.col('std_timegap_minutes_28d') / 
         (pl.col('avg_timegap_minutes_28d') + 1.0))
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias('timegap_consistency_28d'),
        
        ((pl.col('minutes_since_last_txn') - pl.col('avg_timegap_minutes_28d')) /
         (pl.col('avg_timegap_minutes_28d') + 1.0))
        .cast(pl.Float32)
        .alias('timegap_acceleration')
    ])
    
    return df


def compute_velocity_metrics(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute transaction velocity - LAZY VERSION.
    """
    df = df.with_columns([
        pl.col('Amount Paid')
            .rolling_sum(window_size=100)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('daily_amount_paid')
    ])
    
    # Delta and pct change using shift
    df = df.with_columns([
        (pl.col('daily_amount_paid') - 
         pl.col('daily_amount_paid').shift(1).over('Account_HASHED'))
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias('txn_velocity_daily_delta'),
        
        ((pl.col('daily_amount_paid') - 
          pl.col('daily_amount_paid').shift(1).over('Account_HASHED')) /
         (pl.col('daily_amount_paid').shift(1).over('Account_HASHED') + 0.000001))
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias('amount_velocity_daily_pct_change'),
    ])
    
    # Second derivative
    df = df.with_columns([
        (pl.col('amount_velocity_daily_pct_change') - 
         pl.col('amount_velocity_daily_pct_change').shift(1).over('Account_HASHED'))
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias('amount_acceleration_2nd_order'),
    ])
    
    return df


def compute_concentration_metrics(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Detect volume concentration - LAZY VERSION.
    """
    # Use rolling n_unique on integer windows instead of groupby_dynamic
    # Approximate 7d/28d with row windows (assuming ~17 txns/day avg)
    
    df = df.with_columns([
        pl.col('Account_duplicated_0')
            .rolling_n_unique(window_size=120)  # ~7 days
            .over('Account_HASHED')
            .cast(pl.UInt32)
            .alias('unique_counterparties_7d'),
            
        pl.col('Account_duplicated_0')
            .rolling_n_unique(window_size=500)  # ~28 days
            .over('Account_HASHED')
            .cast(pl.UInt32)
            .alias('unique_counterparties_28d'),
    ])
    
    # HHI approximation
    df = df.with_columns([
        ((pl.col('unique_counterparties_28d') + 1) ** (-1.0))
        .cast(pl.Float32)
        .alias('perfect_hhi_28d')
    ])
    
    # Amount stats
    df = df.with_columns([
        pl.col('Amount Paid')
            .rolling_mean(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .cast(pl.Float32)
            .alias('mean_amount_paid_28d_roll'),
            
        pl.col('Amount Paid')
            .rolling_std(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .cast(pl.Float32)
            .alias('std_amount_paid_28d_roll'),
    ])
    
    # CV
    df = df.with_columns([
        (pl.col('std_amount_paid_28d_roll') / 
         (pl.col('mean_amount_paid_28d_roll') + 0.000001))
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias('amount_concentration_cv_28d')
    ])
    
    return df


def compute_round_number_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Detect structuring - LAZY VERSION.
    """
    df = df.with_columns([
        ((pl.col('Amount Paid') % 100.0) == 0.0).cast(pl.Int8).alias('is_multiple_100'),
        ((pl.col('Amount Paid') % 500.0) == 0.0).cast(pl.Int8).alias('is_multiple_500'),
        ((pl.col('Amount Paid') % 1000.0) == 0.0).cast(pl.Int8).alias('is_multiple_1000'),
    ])
    
    df = df.with_columns([
        pl.col('is_multiple_100')
            .rolling_mean(window_size=50)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('round_100_ratio_50txns'),
            
        pl.col('is_multiple_1000')
            .rolling_mean(window_size=50)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('round_1000_ratio_50txns'),
            
        (pl.col('is_multiple_100').rolling_sum(window_size=5) / 5.0)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('consecutive_round_density_5txn'),
    ])
    
    return df


def compute_anomaly_cascade_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Multi-signal anomaly detection - LAZY VERSION.
    """
    df = df.with_columns([
        (pl.col('burst_score_1h') > 2.0).cast(pl.Int8).alias('flag_high_burst'),
        (pl.col('minutes_since_last_txn') > 1440.0).cast(pl.Int8).alias('flag_large_gap'),
        (pl.col('timegap_consistency_28d') < 0.2).cast(pl.Int8).alias('flag_extreme_consistency'),
        (pl.col('amount_concentration_cv_28d') > 1.5).cast(pl.Int8).alias('flag_high_concentration'),
        (pl.col('round_1000_ratio_50txns') > 0.7).cast(pl.Int8).alias('flag_heavy_structuring'),
    ])
    
    df = df.with_columns([
        (pl.col('flag_high_burst') + 
         pl.col('flag_large_gap') + 
         pl.col('flag_extreme_consistency') + 
         pl.col('flag_high_concentration') + 
         pl.col('flag_heavy_structuring'))
        .cast(pl.Int8)
        .alias('anomaly_cascade_score')
    ])
    
    df = df.with_columns([
        (pl.col('anomaly_cascade_score') >= 2).cast(pl.Int8)
            .rolling_mean(window_size=500)
            .over('Account_HASHED')
            .cast(pl.Float32)
            .alias('cascade_frequency_28d')
    ])
    
    return df


def add_advanced_rolling_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add all advanced rolling features - LAZY VERSION.
    
    CRITICAL: Input MUST be sorted by ['Account_HASHED', 'Timestamp'].
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Verify lazy
    if not isinstance(df, pl.LazyFrame):
        raise TypeError("add_advanced_rolling_features REQUIRES LazyFrame. Use streaming checkpoint instead.")
    
    logger.info("  Adding burst scores...")
    df = compute_burst_score(df)
    
    logger.info("  Adding time-gap statistics...")
    df = compute_timegap_statistics(df)
    
    logger.info("  Adding velocity metrics...")
    df = compute_velocity_metrics(df)
    
    logger.info("  Adding concentration metrics...")
    df = compute_concentration_metrics(df)
    
    logger.info("  Adding round number patterns...")
    df = compute_round_number_patterns(df)
    
    logger.info("  Adding anomaly cascade features...")
    df = compute_anomaly_cascade_features(df)
    
    return df