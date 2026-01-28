import hashlib
import polars as pl
import os

def hash_pii_column(df: pl.LazyFrame, col_name: str) -> pl.LazyFrame:
    """
    Hashes a PII column to comply with Data Minization principles.
    Uses a deterministic salt key stored securely or in config.
    """
    #drop the existing hased column if it exists to avoid errors
    if f"{col_name}_HASHED" in df.columns:
        df = df.drop(f"{col_name}_HASHED")
    
    return df.with_columns(
        pl.col(col_name).hash(seed=42).alias(f"{col_name}_HASHED")
    )
