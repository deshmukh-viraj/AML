import hashlib
import polars as pl
import os

def hash_pii_column(df: pl.LazyFrame, col_name: str) -> pl.LazyFrame:
    """
    Hashes a PII column to comply with Data Minization principles.
    Uses a deterministic salt key stored securely or in config.
    """
    # In production, 'PII_SALT' should be an environment variable or secrets manager
    salt = os.getenv("PII_SALT", "DEFAULT_SALT_INSECURE_REPLACE_ME")
    
    return df.with_columns([
        pl.col(col_name).map_elements(
            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest(),
            return_dtype=pl.Utf8
        ).alias(f"{col_name}_HASHED")
    ])
