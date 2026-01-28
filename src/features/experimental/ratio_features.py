import polars as pl

def compute_advanced_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    1.inflow/outflow ratio (pass through detection)
    2.counterparty diversity
    3.sequence contect
    """

    #1. inflow/outflow ratio
    #a ratio close to 1.0 indicates pass-through behavior (smurfing/mule)
    #used a small epsilon to avoid division by zero
    df = df.with_columns([
        (pl.col('total_amount_received_28d') / (pl.col('total_amount_paid_28d') + 1e-6)).alias('inflow_outflow_ratio_28d')
    ])

    #2. counterparty diversity
    #calculating distinct counts in rolling windows in expensive
    #instead, we calculate the number of 'switches' between banks
    #if A->A->B->A, we have 2 switches (A to B, B to A)
    #Calculate a binary flag: 1 if current 'To Bank is different from previous 'To Bank', else 0

    df = df.with_columns([
        (pl.col('To Bank').neq(pl.col('To Bank').shift(1).over('Account_HASHED'))
         .fill_null(0)
         .cast(pl.Int8)
         .alias('is_counterparty_switch'))
    ])

    #rolling sum of switches gives us the diversity of the accounts connections
    df = df.with_columns([
        pl.col('is_counterparty_switch').rolling_sum_by(
            by='Timestamp', window_size='7d'
        ).over('Account_HASHED').shift(1).fill_null(0).alias('counterparty_diversity_7d'),

        pl.col('is_counterparty_switch').rolling_sum_by(
            by='Timestamp', window_size='28d'
        ).over('Account_HASHED').shift(1).fill_null(0).alias('counterparty_diversity_28d')
    ])

    #3. sequence modeling features (contexual lags)
    #provide th model with the context of the previous transaction
    #this helps detect patterns like deposit -> big withdraw -> deposit

    df = df.with_columns([
        pl.col('Amount Paid').shift(1).over('Account_HASHED').alias('prev_amount_paid')

        # # We create a binary "Rush" flag: transaction happened < 1 minute after previous
        ((pl.col('Timestamp') - pl.col('Timestamp').shift(1).over('Account_HASHED')).dt.total_seconds() < 60)
        .fill_null(False)
        .cast(pl.Int8)
        .alias('is_rush_txn')

        #direction change: did we swiithc from senfing to receiving
        (pl.col('Amount Paid') > 0).shift(1).over('Account_HASHED').alias('prev_was_sender')
    ])

    return df