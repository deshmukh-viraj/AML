from pathlib import Path
import polars as pl
import numpy as np

def add_cyclical_time_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add cyclical encodings for temporal features.
    """
    return df.with_columns([
        (2 * np.pi * pl.col('Timestamp').dt.hour() / 24).sin().alias('hour_sin'),
        (2 * np.pi * pl.col('Timestamp').dt.hour() / 24).cos().alias('hour_cos'),
        (2 * np.pi * pl.col('Timestamp').dt.weekday() / 7).sin().alias('day_of_week_sin'),
        (2 * np.pi * pl.col('Timestamp').dt.weekday() / 7).cos().alias('day_of_week_cos'),
        (2 * np.pi * pl.col('Timestamp').dt.day() / 31).sin().alias('day_of_month_sin'),
        (2 * np.pi * pl.col('Timestamp').dt.day() / 31).cos().alias('day_of_month_cos'),
    ])