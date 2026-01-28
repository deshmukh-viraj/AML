import polars as pl

def add_account_lifecycle_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add features tracking account age, tenure, and activity patterns.
    """


    #use window function instead of separate group_by + join
    df = df.with_columns([
        pl.col('Timestamp').min().over('Account_HASHED').alias('account_first_txn')
    ])

    df = df.with_columns([
        #days since account's first transaction
        ((pl.col('Timestamp') - pl.col('account_first_txn'))
         .dt.total_seconds() / 86400)
        .alias('account_tenure_days'),

        # Transaction sequence number for this account
        pl.col('Timestamp')
            .rank(method='ordinal')
            .over('Account_HASHED')
            .alias('txn_rank_in_account_history'),

        # Days since previous transaction
        ((pl.col('Timestamp') - pl.col('Timestamp').shift(1).over('Account_HASHED'))
         .dt.total_seconds() / 86400)
        .fill_null(0)
        .alias('days_since_last_txn'),

        # Flags for account maturity
        (pl.col('Timestamp') - pl.col('account_first_txn') >= pl.duration(days=7))
        .cast(pl.Int8)
        .alias('has_7d_history'),

        (pl.col('Timestamp') - pl.col('account_first_txn') >= pl.duration(days=28))
        .cast(pl.Int8)
        .alias('has_28d_history'),
    ])

    return df