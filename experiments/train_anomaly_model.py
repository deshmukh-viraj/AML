"""
Anomaly Detection Training — Isolation Forest for AML

what this script does:
    trains an Isolation Forest on transaction features and adds an anomaly_score
    column to the train/val/test parquets. That score becomes a feature in the
    main supervised AML classifier downstream.

why isolation forest?
    we don't have reliable "this is definitely money laundering" labels for every
    transaction, and even if we did, we want an independent signal. IF finds
    transactions that look weird compared to everything else — no labels needed.

running it:
    python train_anomaly_model.py

what you get:
    models/anomaly_detector.pkl  — the trained model
    data/processed_with_anomaly/*.parquet — same files but with anomaly_score added
    MLflow run on DagsHub  — params + model artifact logged
"""

import logging
import os
import pickle
import sys
import gc
from pathlib import Path
from typing import List

import mlflow
import dagshub
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


#paths
INPUT_DIR = Path("aml_features")
OUTPUT_DIR = Path("data/processed_with_anomaly")
MODEL_DIR = Path("models")

RANDOM_STATE = 42
CONTAMINATION = "auto"

#we don't need all 25.5M rows to train a good IF model — it subsamples internally
TRAIN_SAMPLE_SIZE = 2_500_000
#controls memory during the scoring loop. 15k rows at a time is safe even
#on a machine with 8GB RAM. Bump it up if scoring is too slow and you have headroom.
CHUNK_SIZE = 100_000

#folumns we never want the model to see.
#-Is Laundering: thats the label, putting it in would be cheating
#-account/tx ids: just row identifiers, meaningless as features
#-timestamps: would let the model learn time patterns, not anomaly patterns
EXCLUDE_COLS = [
    "Is Laundering", "is_laundering",
    "Account_HASHED", "Account", "account_id",
    "transaction_id", "tx_id",
    "Timestamp", "timestamp", "Date", "date",
]

def setup_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folders — {OUTPUT_DIR}, {MODEL_DIR}")


def get_feature_columns(split_name: str) -> List[str]:
    """
    figures out which columns are actually usable features, without reading any rows.

    the second filter is important — sometimes columns like payment_type or
    currency_code slip through and cause a crash downstream.
    """
    path = INPUT_DIR / f"{split_name}_features.parquet"
    if not path.exists():
        logger.error(f"Can't find {path} — check your INPUT_DIR setting")
        sys.exit(1)

    #scan_parquet just reads the file metadata, not the actual rows
    schema = pl.scan_parquet(path).collect_schema()

    numeric_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
        pl.Boolean,
    )

    feature_cols = []
    excluded_by_name= []
    excluded_by_type= []

    for col, dtype in schema.items():
        if any(excl.lower() in col.lower() for excl in EXCLUDE_COLS):
            excluded_by_name.append(col)
        elif not isinstance(dtype, numeric_types):
            #non-numeric
            excluded_by_type.append(f"{col} ({dtype})")
        else:
            feature_cols.append(col)

    logger.info(
        f"Found {len(feature_cols)} usable features out of {len(schema)} total columns "
        f"({len(excluded_by_name)} removed by name, {len(excluded_by_type)} removed by type)"
    )
    return feature_cols


def compute_train_medians(feature_cols: List[str], target_col: str) -> np.ndarray:
    """
    gets the median value of each feature column, using only the training data.

    we need this to fill in NaN/Inf values later before scoring. median makes
    more sense than mean here because transaction amounts are heavily skewed —
    a handful of massive transfers would drag the mean way up and give us a
    poor imputation value for normal transactions.
    """
    path = INPUT_DIR / "train_features.parquet"
    logger.info("Computing medians from training data")

    medians_df = (
        pl.scan_parquet(path)
        .filter(pl.col(target_col)==0)
        .select([pl.col(c).median() for c in feature_cols])
        .collect()
    )

    median_values = medians_df.to_numpy().flatten().astype(np.float32)
    del medians_df
    gc.collect()

    #if a column was entirely null, nanmedian gives Nan just use 0 as fallback
    nan_cols = int(np.isnan(median_values).sum())
    if nan_cols > 0:
        logger.warning(f"{nan_cols} columns were fully null their median defaulted to 0")
    median_values = np.nan_to_num(median_values, nan=0.0)

    zero_cols = int((median_values== 0).sum())
    logger.info(f"Medians ready — {zero_cols} columns have zero median (fine for binary flags)")
    return median_values


def clean_chunk(chunk_df: pl.DataFrame, feature_cols: List[str], medians: np.ndarray) -> np.ndarray:
    """
    takes a small slice of the data and turns it into a numpy array the model can use.

    this runs on every chunk during scoring and on the training sample during fit,
    so it's never touching the full dataset at once.
    """
    X =chunk_df.select(feature_cols).to_numpy().astype(np.float32)

    X[~np.isfinite(X)] = np.nan

    for col_idx in range(X.shape[1]):
        row_indices = np.where(np.isnan(X[:, col_idx]))[0]
        if len(row_indices) > 0:

        #nan_mask = np.isnan(X)
        #if nan_mask.any():
        # np.take picks the right median for each NaN position based on column index
            X[row_indices, col_idx] = medians[col_idx]

    return X


def train_model(feature_cols: List[str], medians: np.ndarray, target_col: str) -> IsolationForest:
    """
    fits the Isolation Forest on a random sample of the training data.

    we dont need all the rows to get a good model here. IF works by building
    random trees on random subsets of data training on sampled rows vs 5M rows
    produces nearly identical anomaly scores in practice, and avoids the OOM
    crash that happened when we tried to load everything.
    """
    path = INPUT_DIR / "train_features.parquet"
    #load legit rows only, feaud rows never touch the model
    legit_df = (
        pl.scan_parquet(path)
        .select(feature_cols + [target_col])
        .filter(pl.col(target_col) ==0)
        .select(feature_cols)
        .collect()
    )
    n_legit = len(legit_df)
    logger.info(f"Lefit rows availble for fitting: {n_legit}")

    #sample fron only legit rows only
    sample_size = min(TRAIN_SAMPLE_SIZE, n_legit)
    rng = np.random.default_rng(RANDOM_STATE)
    row_indices =sorted(rng.choice(n_legit, size=sample_size, replace=False).tolist())
    legit_df =legit_df[row_indices]

    mem_mb =legit_df.estimated_size() / 1024 ** 2
    logger.info(f"Sample loaded — {legit_df.shape[0]:,} rows x {legit_df.shape[1]} features ({mem_mb:.1f} MB)")

    X_train =clean_chunk(legit_df, feature_cols, medians)

    #free the dataframe now
    del legit_df
    gc.collect()
    logger.info("Train DataFrame freed, starting model fit")

    model = IsolationForest(
        n_estimators=200,           
        max_samples=10_050,          
        contamination=CONTAMINATION, 
        max_features=1.0,            
        bootstrap=False,             
        n_jobs=-1,               
        random_state=RANDOM_STATE,
        verbose=1,
    )

    model.fit(X_train)
    logger.info("Fit complete")

    del X_train
    gc.collect()
    return model


def score_and_save(
    model: IsolationForest,
    split_name: str,
    feature_cols: List[str],
    medians: np.ndarray,
    output_path: Path,
):
    """
    sscores one full split (train/val/test) and writes the result to parquet.

    rhe reason we do this in chunks is that even a single split can be millions
    of rows. Converting all of them to a float32 numpy array at once is exactly
    what was causing the OOM crashes 5M rows x 150 features x 4 bytes = ~3GB
    in one shot. Processing 15k rows at a time keeps us well under 10MB per chunk.

    one note on the score: score_samples() returns more negative values for
    anomalies (shorter isolation path =easier to isolate = more anomalous).
    we flip the sign so higher score =more suspicious, which is easier to
    reason about when reviewing flagged transactions.
    """
    input_path = INPUT_DIR / f"{split_name}_features.parquet"

    n_rows   = pl.scan_parquet(input_path).select(pl.len()).collect().item()
    n_chunks = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info(f"Scoring {split_name} — {n_rows:,} rows across {n_chunks} chunks")

    lazy_reader = pl.scan_parquet(input_path)
    parquet_writer = None
    all_scores = []

    for chunk_idx, row_start in enumerate(range(0, n_rows, CHUNK_SIZE)):

        chunk_df =lazy_reader.slice(row_start, CHUNK_SIZE).collect()

        X = clean_chunk(chunk_df, feature_cols, medians)
        scores = -model.score_samples(X)   
        del X
        gc.collect()

        #polars dataframes are immutable, so with_columns returns a new one
        chunk_df =chunk_df.with_columns(pl.Series("anomaly_score", scores))
        arrow_chunk = chunk_df.to_arrow()

        if parquet_writer is None:
            parquet_writer =pq.ParquetWriter(
                str(output_path), arrow_chunk.schema, compression="snappy"
            )
        parquet_writer.write_table(arrow_chunk)

        all_scores.append(scores)
        del chunk_df, arrow_chunk, scores
        gc.collect()

        if (chunk_idx + 1) % 10 ==0 or (chunk_idx + 1) == n_chunks:
            logger.info(
                f"  {split_name} — {chunk_idx + 1}/{n_chunks} chunks done "
                f"(up to row {min(row_start + CHUNK_SIZE, n_rows):,})"
            )

    if parquet_writer:
        parquet_writer.close()

    scores_arr = np.concatenate(all_scores)
    del all_scores
    gc.collect()

    #log the distribution — useful later when picking a threshold for flagging
    logger.info(
        f"Score distribution [{split_name}] — "
        f"mean: {scores_arr.mean():.4f}  std: {scores_arr.std():.4f}  "
        f"min: {scores_arr.min():.4f}  max: {scores_arr.max():.4f}  "
        f"p95: {np.percentile(scores_arr, 95):.4f}  p99: {np.percentile(scores_arr, 99):.4f}"
    )

    del scores_arr
    gc.collect()

    file_mb = os.path.getsize(output_path) / 1024 ** 2
    logger.info(f"  Written to {output_path} ({n_rows:,} rows, {file_mb:.1f} MB)")


def save_model(model: IsolationForest):
    """saves the model plus the settings used to train it, so we can reproduce later."""
    model_path = MODEL_DIR / "anomaly_detector.pkl"

    with open(model_path, "wb") as f:
        pickle.dump({
            "model" : model,
            "contamination":CONTAMINATION,
            "random_state" : RANDOM_STATE,
            "train_sample_size":TRAIN_SAMPLE_SIZE,
        }, f)

    size_mb = os.path.getsize(model_path) / 1024 ** 2
    logger.info(f"Model saved to {model_path} ({size_mb:.1f} MB)")


def log_to_mlflow(n_features: int):
    """logs the run params and model file to MLflow on DagsHub."""
    logger.info("Logging to MLflow")

    mlflow.set_tracking_uri("https://dagshub.com/virajdeshmukh080818/AML.mlflow")
    dagshub.init(repo_owner="virajdeshmukh080818", repo_name="AML", mlflow=True)
    mlflow.set_experiment("Unsupervised_Anomaly_Feature_Gen")

    with mlflow.start_run(run_name="isolation_forest_training"):
        mlflow.log_params({
            "model_type" : "IsolationForest",
            "n_estimators" :200,
            "contamination" : CONTAMINATION,
            "max_samples" : "10_050",
            "max_features" : 1.0,
            "bootstrap" : False,
            "random_state" : RANDOM_STATE,
            "n_features" : n_features,
            "train_sample_size": TRAIN_SAMPLE_SIZE,
            "chunk_size" : CHUNK_SIZE,
        })
        mlflow.log_artifact(str(MODEL_DIR / "anomaly_detector.pkl"))

    logger.info("MLflow logging done")


def main():
    logger.info("=" * 60)
    logger.info("Starting anomaly detection training")
    logger.info(f"Input : {INPUT_DIR}")
    logger.info(f"Output : {OUTPUT_DIR}")
    logger.info(f"Train sample : {TRAIN_SAMPLE_SIZE:,} rows")
    logger.info(f"Chunk size : {CHUNK_SIZE:,} rows")
    logger.info("=" * 60)

    setup_directories()

    #read column names from schema only — no actual data loaded here
    feature_cols = get_feature_columns("train")
    if not feature_cols:
        logger.error("No usable feature columns found — check EXCLUDE_COLS")
        sys.exit(1)

    # quick fraud rate check using lazy aggregation — again, no full load
    train_path = INPUT_DIR / "train_features.parquet"
    schema = pl.scan_parquet(train_path).collect_schema()
    target_col = "Is Laundering" 

    if target_col in schema:
        stats = (
            pl.scan_parquet(train_path)
            .select([
                pl.col(target_col).mean().alias("rate"),
                pl.col(target_col).sum().alias("fraud"),
                pl.len().alias("total"),
            ])
            .collect()
        )
        logger.info(
            f"Fraud rate in train: {stats['rate'].item():.4%} "
            f"({int(stats['fraud'].item()):,} out of {int(stats['total'].item()):,})"
        )
        del stats
        gc.collect()
    else:
        logger.warning("Couldnt find target column skipping fraud rate check")

    #compute medians lazily only scalar results come back into memory
    medians = compute_train_medians(feature_cols, target_col)

    #train on a random sample to keep RAM usage reasonable
    model = train_model(feature_cols, medians, target_col)

    #save before scoring so the model is safe even if scoring fails halfway
    save_model(model)

    output_paths = {
        "train": OUTPUT_DIR / "train_features.parquet",
        "val": OUTPUT_DIR / "val_features.parquet",
        "test" : OUTPUT_DIR / "test_features.parquet",
    }

    for split_name in ["train", "val", "test"]:
        logger.info("─" * 60)
        score_and_save(model, split_name, feature_cols, medians, output_paths[split_name])

    log_to_mlflow(len(feature_cols))

    logger.info("=" * 60)
    logger.info("Done")
    logger.info(f" Model : {MODEL_DIR / 'anomaly_detector.pkl'}")
    logger.info(f" Scored data : {OUTPUT_DIR}")
    logger.info(f" Features : {len(feature_cols)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()