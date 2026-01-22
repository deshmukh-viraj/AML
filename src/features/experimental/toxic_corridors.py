import polars as pl

def apply_toxic_corridor_features(df: pl.LazyFrame, toxic_corridors: pl.DataFrame) -> pl.LazyFrame:
  """
  Add features measuring exposure to toxic corridors.
  """
  #join with toxic corridor list
  df = df.join(
      toxic_corridors.lazy(),
      on=['From Bank', 'To Bank'],
      how='left'
  ).with_columns([
      pl.col('is_toxic_corridor').fill_null(0),
      pl.col('fraud_rate').fill_null(0).alias('corridor_risk_score'),
  ])

  df = df.with_columns([
      pl.col('is_toxic_corridor')
          .rolling_sum_by(window_size='28d', by='Timestamp')
          .over('Account')
          .shift(1)
          .fill_null(0)
          .alias('toxic_corridor_count_28d'),

      (pl.col('Amount Paid') * pl.col('is_toxic_corridor'))
          .rolling_sum_by(window_size='28d', by='Timestamp')
          .over('Account')
          .shift(1)
          .fill_null(0)
          .alias('toxic_corridor_volume_28d'),
  ])

  df = df.with_columns([
      (pl.col('toxic_corridor_volume_28d') /
        pl.when(pl.col('total_amount_paid_28d') > 0)
          .then(pl.col('total_amount_paid_28d'))
          .otherwise(1.0))
      .alias('pct_volume_via_toxic_corridors'),
  ])

  return df
