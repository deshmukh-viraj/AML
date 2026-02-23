# Feature Pipeline Documentation

## Overview

The feature pipeline runs **feature engineering only** (no modeling) on raw AML transaction and account data. It produces train/validation/test feature Parquet files ready for downstream modeling.

**Entry point:** `experiments/run_feature_pipeline.py`

---

## Quick Start

From the project root:

```bash
python experiments/run_feature_pipeline.py \
  --trans-path data/raw/HI-Medium_Trans.csv \
  --accounts-path data/raw/HI-Medium_accounts.csv \
  --output-dir aml_features \
  --sample 0.1
```

---

## CLI Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--trans-path` | Path | `data/raw/HI-Medium_Trans.csv` | Path to transactions CSV. Must exist. |
| `--accounts-path` | Path | `data/raw/HI-Medium_accounts.csv` | Path to accounts CSV. Must exist. |
| `--output-dir` | Path | `aml_features` | Directory for output Parquet files. Created if missing. |
| `--sample` | float | `None` | Optional fraction of transactions to use (e.g. `0.1` for 10%). Speeds up experiments. |

**Examples:**

```bash
# Default paths, full data
python experiments/run_feature_pipeline.py

# Custom paths
python experiments/run_feature_pipeline.py --trans-path data/raw/My_Trans.csv --accounts-path data/raw/My_accounts.csv

# Quick run on 1% sample
python experiments/run_feature_pipeline.py --sample 0.01
```

---

## What the Pipeline Does

1. **CSV → Parquet (cached)**  
   Converts transactions and accounts CSVs to Parquet next to the CSVs. Reuses existing Parquet on later runs.

2. **Load & optional sampling**  
   Loads transactions (Polars LazyFrame) and accounts. If `--sample` is set, samples that fraction of transactions (seed=42).

3. **PII hashing**  
   Hashes `Account` into `Account_HASHED` (UTF-8).

4. **Temporal splits**  
   - Test: last 7 days  
   - Val: 7 days before test  
   - Train: all before val  
   Splits are sorted by `Account_HASHED` and `Timestamp`.

5. **Feature engineering (per split, in batches by account)**  
   - Base features (temporal, Benford, lifecycle)  
   - Entity/account stats (from accounts)  
   - Standard rolling features  
   - Ratio and derived features  
   - Advanced rolling (burst, time-gaps, velocity)  
   - Counterparty entropy  
   - Network features  
   - Toxic corridor features  

6. **Validation**  
   Basic checks on train features: row/column counts, missing values in critical features, class distribution (if `Is Laundering` present).

7. **Output**  
   Writes `train_features.parquet`, `val_features.parquet`, `test_features.parquet` into `--output-dir`.

---

## Output Files

| File | Description |
|------|-------------|
| `train_features.parquet` | Training split features |
| `val_features.parquet` | Validation split features |
| `test_features.parquet` | Test split features |

These can be loaded with Polars or pandas and fed into `src/modeling.py`, `src/balancing.py`, or the orchestrator.

---

## Implementation Notes

- **Script role:** `run_feature_pipeline.py` is a thin CLI wrapper; it calls `src.features.build_features.build_all_features()` with `compute_anomaly_scores=False`.
- **Run from project root** so default paths and `src` imports resolve correctly.
- **First run** may be slower due to CSV→Parquet conversion; later runs use cached Parquet.
- **Memory:** Splits are processed in account-based batches (default 15,000 accounts per batch) with streaming collect to limit memory use.

---

## Related Code

| Component | Path | Role |
|-----------|------|------|
| CLI script | `experiments/run_feature_pipeline.py` | Parses args, validates paths, calls `build_all_features` |
| Feature builder | `src/features/build_features.py` | `build_all_features`, `build_training_features`, batch processing |
| Config | `config.py` | Data/model settings (e.g. `USE_POLARS`, `CHUNK_SIZE`) |
