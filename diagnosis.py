"""
quick_diagnostic.py
Run this on the already-saved anomaly output parquets.
Does not retrain or rescore anything.
"""

import polars as pl
from pathlib import Path

OUTPUT_DIR = Path("data/processed_with_anomaly")
TARGET_COL = "Is Laundering"

for split in ["train", "val", "test"]:
    path = OUTPUT_DIR / f"{split}_features.parquet"

    stats = (
        pl.scan_parquet(path)
        .select([
            pl.len().alias("total_rows"),
            pl.col(TARGET_COL).sum().alias("fraud_count"),
            pl.col(TARGET_COL).mean().alias("fraud_rate"),
            pl.col("anomaly_score")
              .filter(pl.col(TARGET_COL) == 1)
              .mean().alias("fraud_score_mean"),
            pl.col("anomaly_score")
              .filter(pl.col(TARGET_COL) == 0)
              .mean().alias("legit_score_mean"),
            pl.col("anomaly_score")
              .filter(pl.col(TARGET_COL) == 1)
              .max().alias("fraud_score_max"),
            pl.col("anomaly_score").max().alias("overall_max"),
        ])
        .collect()
    )

    fraud_count      = int(stats["fraud_count"].item())
    total_rows       = int(stats["total_rows"].item())
    fraud_rate       = float(stats["fraud_rate"].item())
    fraud_mean       = float(stats["fraud_score_mean"].item())
    legit_mean       = float(stats["legit_score_mean"].item())
    fraud_max        = float(stats["fraud_score_max"].item())
    overall_max      = float(stats["overall_max"].item())
    separation       = fraud_mean - legit_mean

    print(f"\n{'='*55}")
    print(f"Split        : {split}")
    print(f"Total rows   : {total_rows:,}")
    print(f"Fraud count  : {fraud_count:,} ({fraud_rate:.4%})")
    print(f"Legit mean   : {legit_mean:.4f}")
    print(f"Fraud mean   : {fraud_mean:.4f}")
    print(f"Separation   : {separation:.4f}  (want > 0.05)")
    print(f"Fraud max    : {fraud_max:.4f}")
    print(f"Overall max  : {overall_max:.4f}")
    print(f"{'='*55}")