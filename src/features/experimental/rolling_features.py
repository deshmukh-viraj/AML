import polars as pl

def compute_rolling_features_batch1(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Batch 1: Transaction counts only
    """
    return df.with_columns([
        pl.col('Timestamp')
            .rolling_count_by(period='1h', by='Timestamp')
            .over('Account')
            .shift(1)
            .fill_null(0)
            .alias('txn_count_1h'),

        pl.col('Timestamp')
            .rolling_count_by(period='24h', by='Timestamp')
            .over('Account')
            .shift(1)
            .fill_null(0)
            .alias('txn_count_24h'),

        pl.col('Timestamp')
            .rolling_count_by(period='7d', by='Timestamp')
            .over('Account')
            .shift(1)
            .fill_null(0)
            .alias('txn_count_7d'),

        pl.col('Timestamp')
            .rolling_count_by(period='28d', by='Timestamp')
            .over('Account')
            .shift(1)
            .fill_null(0)
            .alias('txn_count_28d'),
    ])


def compute_rolling_features_batch2(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Batch 2: Volume statistics
    """
    return df.with_columns([
        pl.col('Amount Paid')
            .rolling_sum_by(period='28d', by='Timestamp')
            .over('Account')
            .shift(1)
            .fill_null(0)
            .alias('total_amount_paid_28d'),

        pl.col('Amount Received')
            .rolling_sum_by(period='28d', by='Timestamp')
            .over('Account')
            .shift(1)
            .fill_null(0)
            .alias('total_amount_received_28d'),
    ])


def compute_rolling_features_batch3(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Batch 3: Statistical aggregations
    """
    return df.with_columns([
        pl.col('Amount Paid')
            .rolling_mean_by(period='28d', by='Timestamp')
            .over('Account')
            .shift(1)
            .alias('mean_amount_paid_28d'),

        pl.col('Amount Paid')
            .rolling_std_by(period='28d', by='Timestamp')
            .over('Account')
            .shift(1)
            .alias('std_amount_paid_28d'),

        pl.col('Amount Paid')
            .rolling_median_by(period='28d', by='Timestamp')
            .over('Account')
            .shift(1)
            .alias('median_amount_paid_28d'),

        pl.col('Amount Paid')
            .rolling_max_by(period='28d', by='Timestamp')
            .over('Account')
            .shift(1)
            .alias('max_amount_paid_28d'),
    ])
