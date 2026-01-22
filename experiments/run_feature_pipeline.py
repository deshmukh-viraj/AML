from pathlib import Path
from datetime import timedelta
import logging
import polars as pl
pl.Config.set_tbl_rows(20)
pl.Config.set_streaming_chunk_size(100_000)
import gc

from src.features.experimental.time_features import add_cyclical_time_features
from src.features.experimental.benford_features import add_benford_features
from src.features.experimental.lifecycle_features import add_account_lifecycle_features
from src.features.experimental.rolling_features import (
    compute_rolling_features_batch1,
    compute_rolling_features_batch2,
    compute_rolling_features_batch3,
)
from src.features.experimental.derived_features import compute_derived_features
from src.features.experimental.toxic_corridors import apply_toxic_corridor_features
from src.utils.hashing import hash_pii_column
from src.config import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


from datetime import timedelta
class AMLFeatureEngineer:
  def __init__(self,
                transactions_path: str,
                accounts_path: str,
                output_dir: str = './features'):
      self.transactions_path = transactions_path
      self.accounts_path = accounts_path
      self.output_dir = Path(output_dir)
      self.output_dir.mkdir(exist_ok=True)
      self.toxic_corridors = None

  def load_data(self) -> tuple[pl.LazyFrame, pl.DataFrame]:
      """
      Load and validate raw data.
      """
      print(" Loading data...")

      #Keep transactions lazy for memory efficiency
      trans = pl.scan_csv(
          self.transactions_path,
          try_parse_dates=True,
          low_memory=True
      )

      #accounts is small, can be eager
      accounts = pl.scan_csv(self.accounts_path).collect()

      #get count without collecting entire dataset
      trans_count = trans.select(pl.count()).collect()[0, 0]

      print(f" ✓ Loaded {trans_count:,} transactions (lazy)")
      print(f" ✓ Loaded {len(accounts):,} accounts")

      return trans, accounts

  def create_temporal_split(self, df: pl.LazyFrame) -> pl.LazyFrame:
      """
      Create train/val/test split based on time.
      # OPTIMIZED: Works with LazyFrame, defers execution
      """
      print(" Creating temporal splits...")

      #get max timestamp
      max_timestamp = df.select(pl.col('Timestamp').max()).collect()[0, 0]

      test_start = max_timestamp - timedelta(days=7)
      val_start = test_start - timedelta(days=7)

      df = df.with_columns([
          pl.when(pl.col('Timestamp') < val_start)
            .then(pl.lit('train'))
            .when(pl.col('Timestamp') < test_start)
            .then(pl.lit('val'))
            .otherwise(pl.lit('test'))
            .alias('split')
      ])


      split_counts = df.group_by('split').agg(pl.count().alias('count')).collect()

      for row in split_counts.iter_rows(named=True):
          print(f" ✓ {row['split'].capitalize()}: {row['count']:,}")

      return df

  def split_and_save_raw(self, df: pl.LazyFrame):
      '''
      split and save raw data
      this allows processiing each split independently
      '''
      print('Splitting and saving raw data....')

      for split_name in ['train', 'val', 'test']:
        print(f" Saving raw {split_name} split....")
        output_path = self.output_dir / f"{split_name}_raw.parquet"
        (
            df.filter(pl.col('split') == split_name)
            .sink_parquet(output_path)
        )

        print(f"  Saved {split_name}_raw.parquet")


  def process_split(self, split_name: str, accounts: pl.DataFrame, toxic_corridors: pl.DataFrame = None):
      """
      Process a single split with all features.
      """
      logger.info(f"{'='*60}")
      logger.info(f"Processing {split_name.upper()} split...")
      logger.info(f"{'='*60}")

      #Load Split
      input_path = self.output_dir / f'{split_name}_raw.parquet'
      df = pl.scan_parquet(input_path)

      logger.info("  Phase 0: Hashing PII (acoount IDs..)")
      df = hash_pii_column(df, 'Account')

      #Basic Features (Map operations, low memory)
      logger.info("  Phase 1: Basic time & benford features...")
      df = add_cyclical_time_features(df)
      df = add_benford_features(df)

      #Entity Join (Must happen before sort)
      logger.info("  Phase 2: Entity enrichment...")
      df = self._add_entity_features(df, accounts)

      #ACCOUNT LIFECYCLE
      #e sort here ONCE. This sort persists through the lazy frame evaluation.
      logger.info("  Phase 3: Sorting by Account & Timestamp for rolling windows...")
      df = df.sort(['Account_HASHED', 'Timestamp'])

      logger.info("  Phase 3b: Calculating Lifecycle features...")
      df = add_account_lifecycle_features(df)

      #CHECKPOINT 1
      temp_path = self.output_dir / f'{split_name}_checkpoint1.parquet'
      logger.info(f"  Saving Checkpoint 1 -> {temp_path.name}")
      df.sink_parquet(temp_path)
      df = pl.scan_parquet(temp_path)

      #ROLLING BATCH 1 (Counts)
      logger.info("  Phase 4a: Rolling counts...")
      df = compute_rolling_features_batch1(df)

      #CHECKPOINT 2
      temp_path = self.output_dir / f'{split_name}_checkpoint2.parquet'
      logger.info(f"  Saving Checkpoint 2 -> {temp_path.name}")
      df.sink_parquet(temp_path)
      df = pl.scan_parquet(temp_path)

      #ROLLING BATCH 2 (Volumes)
      logger.info("  Phase 4b: Rolling volumes...")
      df = compute_rolling_features_batch2(df)

      #vHECKPOINT 3
      temp_path = self.output_dir / f'{split_name}_checkpoint3.parquet'
      logger.info(f"  Saving Checkpoint 3 -> {temp_path.name}")
      df.sink_parquet(temp_path)

      #if this is the training split, we compute toxic corridors now
      # using this checkpoint, as requested by the user logic.
      if split_name == 'train' and self.toxic_corridors is None:
          logger.info("  -> Computing toxic corridors from Checkpoint 3...")
          self.compute_toxic_corridors(temp_path)

      #reload for next phase
      df = pl.scan_parquet(temp_path)

      #ROLLING BATCH 3 (Statistics)
      logger.info("  Phase 4c: Rolling statistics (Mean, Std, Median)...")
      try:
          df = compute_rolling_features_batch3(df)
      except Exception as e:
          logger.error(f"CRITICAL: Failed during rolling stats (likely OOM): {e}")
          raise

      #DERIVED FEATURES
      logger.info("  Phase 5: Derived features...")
      df = compute_derived_features(df)

      #TOXIC CORRIDORS
      if toxic_corridors is not None:
          logger.info("  Phase 6: Toxic corridor features...")
          df = apply_toxic_corridor_features(df, toxic_corridors)

      # FINAL OUTPUT
      output_path = self.output_dir / f'{split_name}_features.parquet'
      logger.info(f"  Saving Final Output -> {output_path.name}")
      df.sink_parquet(output_path)

      # CLEANUP CHECKPOINTS
      self._cleanup_checkpoints(split_name)

      logger.info(f"   {split_name.upper()} complete!")


  def _add_entity_features(self, df: pl.LazyFrame, accounts: pl.DataFrame) -> pl.LazyFrame:
      """
      Join with accounts and add entity-level features.
      """

      entity_stats = (
          accounts.lazy().group_by('Entity ID')
          .agg([
              pl.count().alias('entity_account_count'),
              pl.col('Bank ID').n_unique().alias('entity_bank_count'),
          ])
      )

      #convert to lazy for joining
      df = (
          df.join(
              accounts.select(['Account Number', 'Entity ID']).lazy(),
              left_on='Account',
              right_on='Account Number',
              how='left'
          )
          .join(entity_stats.lazy(), on='Entity ID', how='left')
      )

      return df


  def compute_toxic_corridor_features(self, train_path: Path) -> None:
      """
      Add toxic corridor features.
      """
      print('Computing toxic corridors...')
      
      logging.warning("COMPLIANCE CHECK: Toxic Corridors are calculated on the TRAINING SET only ")
     
      #compute corridor aggregates lazily then collect only small result
      toxic_df = (
          pl.scan_parquet(train_path).group_by(['From Bank', 'To Bank']).agg([
              pl.count().alias('total_txns'),
              pl.col('Is Laundering').sum().alias('fraud_txns'),
              pl.col('Amount Paid').sum().alias('total_volume'),
              pl.col('Amount Paid').filter
                  (pl.col('Is Laundering') == 1).sum().alias('fraud_volume')

          ]).with_columns([
              (pl.col('fraud_txns') / pl.col('total_txns')).alias('fraud_rate'),
              (pl.col('fraud_volume') / pl.col('total_volume')).alias('volume_fraud_rate'),

          ]).filter(
              (pl.col('total_txns') >= 1000) & (pl.col('fraud_rate') >= 0.05)
          ).select([
              'From Bank', 'To Bank', 'fraud_rate', 'volume_fraud_rate'
          ]).with_columns([
              pl.lit(1).alias('is_toxic_corridor')
          ]).collect()
      )
      
      #high volume alert
      if toxic_df.height() > 0:
          #check if a single bank dominated toxic volume
          top_risky_bank = toxic_df.group_by('To Bank').agg(
              pl.col('total_txns').sum().alias('toxic_count')
          ).sort('toxic_count', descending=True).row(0)

          if top_risky_bank['toxic_count'] > toxic_df['total_txns'].sum() * 0.10:
              logger.warning(f"AUDIT ALERT: High volume toxic traffic detected to Bank ID {top_risky_bank['To Bank']}")

      print(f" Identified {len(toxic_df)} toxic corridors")
      self.toxic_corridors = toxic_df


  def _cleanup_checkpoints(self, split_name: str):
        """Safely remove checkpoint files."""
        for i in range(1, 4):
            checkpoint = self.output_dir / f'{split_name}_checkpoint{i}.parquet'
            try:
                if checkpoint.exists():
                    checkpoint.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {checkpoint}: {e}")

  def run(self):
        """Execute complete pipeline."""
        logger.info("="*60)
        logger.info("AML FEATURE ENGINEERING PIPELINE (OPTIMIZED)")
        logger.info("="*60)

        #load and Split
        trans, accounts = self.load_data()
        trans = self.create_temporal_split(trans)

        #save Raw Splits (Low memory overhead)
        self.split_and_save_raw(trans)

        #clean up large lazy frame from memory
        del trans
        import gc
        gc.collect()

        #process Training (to get toxic corridors)
        #we pass toxic_corridors=None initially.
        #the logic inside process_split will call compute_toxic_corridors if split == 'train'.
        self.process_split('train', accounts, toxic_corridors=None)

        #reprocess Train with toxic corridor features
        #we reload checkpoint 3, add toxic features, and overwrite final parquet
        logger.info("Re-processing TRAIN with toxic corridor features...")
        train_path = self.output_dir / 'train_checkpoint3.parquet'
        if train_path.exists():
            df = pl.scan_parquet(train_path)
            df = apply_toxic_corridor_features(df, self.toxic_corridors)
            df.sink_parquet(self.output_dir / 'train_features.parquet')
            logger.info("Train re-processing complete.")
        else:
            logger.error("Train checkpoint 3 missing, skipping re-processing.")

        #process Val and Test (with toxic corridors)
        self.process_split('val', accounts, toxic_corridors=self.toxic_corridors)
        self.process_split('test', accounts, toxic_corridors=self.toxic_corridors)

        #final Cleanup
        logger.info("Cleaning up raw intermediate files...")
        for split_name in ['train', 'val', 'test']:
            raw_file = self.output_dir / f'{split_name}_raw.parquet'
            try:
                if raw_file.exists():
                    raw_file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {raw_file}: {e}")

        logger.info("="*60)
        logger.info(" PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Output directory: {self.output_dir.absolute()}")


# USAGE
if __name__ == "__main__":
    # Ensure Kaggle API is setup before running this

    # 1. Setup paths
    # Assuming files are in current directory, adjust if needed
    trans_csv = DATA_DIR / 'HI-Medium_Trans.csv'
    acc_csv = DATA_DIR / 'HI-Medium_accounts.csv'

    # Check if files exist to prevent errors
    if not Path(trans_csv).exists():
        print(f"ERROR: {trans_csv} not found. Please download the dataset first.")
    else:
        pipeline = AMLFeatureEngineer(
            transactions_path=trans_csv,
            accounts_path=acc_csv,
            output_dir='./aml_features'
        )
        pipeline.run()