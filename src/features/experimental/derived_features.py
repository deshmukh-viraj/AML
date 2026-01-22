import polars as pl

def compute_derived_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute derived features from rolling statistics."""
    return df.with_columns([
        #transaction velocity
        (pl.col('txn_count_24h') / (pl.col('txn_count_7d') + 1))
            .alias('txn_velocity_ratio'),
        
        #amount deviation from average
        ((pl.col('Amount Paid') - pl.col('mean_amount_paid_28d')) / 
         (pl.col('mean_amount_paid_28d') + 1)).alias('amount_deviation_from_avg'),
        
        #size ratio to maximum
        (pl.col('Amount Paid') / (pl.col('max_amount_paid_28d') + 1))
            .alias('amount_vs_max_ratio'),
        
        #volume concentration
        (pl.col('total_amount_paid_28d') / (pl.col('txn_count_28d') + 1))
            .alias('avg_txn_size_28d'),
    ])

