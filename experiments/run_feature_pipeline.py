from pathlib import Path
from datetime import timedelta
import logging
import polars as pl
from functools import lru_cache
from typing import Tuple, Dict
from datetime import timedelta
import psutil
import time
import shutil

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
from src.features.experimental.ratio_features import compute_advanced_features
from src.features.experimental.derived_features import compute_derived_features
from src.features.experimental.toxic_corridors import apply_toxic_corridor_features
from src.utils.hashing import hash_pii_column
from src.config import DATA_DIR
from src.features.experimental.precompute_entity_stats import precompute_entity_stats
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ResorceTracker:
    """track CPU and memory usage for each pipleine phase"""
    
    def __init__(self):
        self.metrics = []
        self.process = psutil.Process()
    
    def start_phase(self, phase_name: str):
        '''record start of phase'''
        
        self.current_phase = phase_name
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024**3
        logger.info(f" Starting: {phase_name}")

    def end_phase(self):
        '''record end phase'''
        duration = time.time() - self.start_time
        end_memory = self.process.memory_info().rss / 1024**3
        memory_delta = end_memory - self.start_memory

        metric = {
            'phase': self.current_phase,
            'duration_sec': duration,
            'memory_start_gb': self.start_memory,
            'memory_end_gb': end_memory,
            'memory_delta_gb': memory_delta
        }
        self.metrics.append(metric)

        logger.info(
            f"Completed: {self.current_phase} |"
            f"Time: {duration:.2f}s |"
            f"Memory: {end_memory:.2f}GB ({memory_delta:+.2f}GB)"
        ) 

    def get_summary(self):
        return pl.DataFrame(self.metrics)
        

class AMLFeatureEngineer:
  def __init__(self,
                transactions_path: str,
                accounts_path: str,
                output_dir: str = './features'):
      self.transactions_path = transactions_path
      self.accounts_path = accounts_path
      self.output_dir = Path(output_dir)
      self.output_dir.mkdir(exist_ok=True)
      self.tracker = ResorceTracker()
      self.toxic_corridors = None
      self.accounts_enriched = None

  def load_data(self) -> tuple[pl.LazyFrame, pl.DataFrame]:
      """
      Load and validate raw data.
      """
      self.tracker.start_phase("Data Loading")
      logger.info(" Loading data...")

      scan_schema = {
          "From Bank": pl.Categorical,
          "To Bank": pl.Categorical,
          "Amount Paid": pl.Float32,
          "Amount Received": pl.Float32,
          "Is Laundering": pl.Int8
      }

      #Keep transactions lazy for memory efficiency
      trans = pl.scan_csv(
          self.transactions_path,
          try_parse_dates=True,
          low_memory=True,
          schema_overrides=scan_schema
      )

      #accounts is small, can be eager
      accounts = pl.scan_csv(self.accounts_path).collect()

      #get count without collecting entire dataset
      trans_count = trans.select(pl.len()).collect(engine='streaming')[0, 0]

      print(f"  Loaded {trans_count:,} transactions (lazy)")
      print(f"  Loaded {len(accounts):,} accounts")

      self.accounts_enriched = precompute_entity_stats(accounts)
      self.tracker.end_phase()

      return trans, accounts

  
  def create_temporal_split(self, df: pl.LazyFrame) -> pl.LazyFrame:
      """
      Create train/val/test split based on time.
      """
      self.tracker.start_phase('Tempral split creation')

      logger.info(" Creating temporal splits...")

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


      split_counts = df.group_by('split').agg(pl.len().alias('count')).collect(engine='streaming')

      for row in split_counts.iter_rows(named=True):
          logger.info(f"  {row['split'].capitalize()}: {row['count']:,}")

      self.tracker.end_phase()
      return df

  def split_and_save_raw(self, df: pl.LazyFrame):
      '''
      split and save raw data
      this allows processiing each split independently
      '''
      self.tracker.start_phase('saving raw splits')
      logger.info('Splitting and saving raw data....')

      for split_name in ['train', 'val', 'test']:
        print(f" Saving raw {split_name} split....")
        output_path = self.output_dir / f"{split_name}_raw.parquet"

        if output_path.exists():
            try:
                output_path.unlink()
            except Exception as e:
                logger.warning(f"Could not unlink {output_path}: {e}")
        (
            df.filter(pl.col('split') == split_name)
            .sink_parquet(output_path, compression='zstd', statistics=True)
        )

        logger.info(f"  Saved {split_name}_raw.parquet")
      self.tracker.end_phase()

  def _add_entity_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
      """
      join with pre computed entity features"""
      df = df.join(
          self.accounts_enriched.lazy(),
          left_on='Account',
          right_on='Account Number',
          how='left'
      )
      return df 
  

  def compute_toxic_corridor_features(self, train_df: pl.LazyFrame) -> pl.DataFrame:
      """
      Add toxic corridor features.
      """
      self.tracker.start_phase("Computing toxic corridors")
      logger.info('Computing toxic corridorss....')
      logging.warning("COMPLIANCE CHECK: Toxic Corridors are calculated on the TRAINING SET only ")
     
      #compute corridor aggregates lazily then collect only small result
      toxic_df = (
          train_df.group_by(['From Bank', 'To Bank']).agg([
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
      logger.info(f" Identified {len(toxic_df)} toxic corridors")
      self.tracker.end_phase()
      return toxic_df
      
    
  def process_split(self, split_name: str, toxic_corridors: pl.DataFrame = None):
      """
      Process a single split with all features.
      """
      logger.info(f"{'='*60}")
      logger.info(f"Processing {split_name.upper()} split...")
      logger.info(f"{'='*60}")

      #Load Split
      input_path = self.output_dir / f'{split_name}_raw.parquet'
      df = pl.scan_parquet(input_path)

      df = df.with_columns([
          pl.col('From Bank').cast(pl.Categorical),
          pl.col('To Bank').cast(pl.Categorical),
          pl.col('Amount Paid').cast(pl.Float32),
          pl.col('Amount Received').cast(pl.Float32)
      ])

      # Phase 0: PII Hashing
      self.tracker.start_phase(f"{split_name} - PII Hashing")
      logger.info("  Phase 0: Hashing PII (acoount IDs..)")
      df = hash_pii_column(df, 'Account')
      self.tracker.end_phase()

      #Basic Features (Map operations, low memory)
      self.tracker.start_phase(f"{split_name} - Time & Benford Features")
      logger.info("  Phase 1: Basic time & benford features...")
      df = add_cyclical_time_features(df)
      df = add_benford_features(df)
      self.tracker.end_phase()

      #Entity Join (Must happen before sort)
      self.tracker.start_phase(f"{split_name} - Entity Enrichment")
      logger.info("  Phase 2: Entity enrichment...")
      df = self._add_entity_features(df)

      if 'Account' in df.columns:
          df = df.drop('Account')

      self.tracker.end_phase()

      #ACCOUNT LIFECYCLE
      #e sort here ONCE. This sort persists through the lazy frame evaluation.
      self.tracker.start_phase(f"{split_name} - Sort and Lifecycle")
      logger.info("  Phase 3: Sorting by Account & Timestamp for rolling windows...")
      df = df.sort(['Account_HASHED', 'Timestamp'], maintain_order=True)

      logger.info("  Phase 3b: Calculating Lifecycle features...")
      df = add_account_lifecycle_features(df)
      core_cols = ['Account_HASHED', 'Timestamp', 'Amount Paid', 'Amount Received', 'Is Laundering', 'From Bank', 'To Bank', 'split']
      curr_cols = df.collect_schema().names()
      generated_cols = [c for c in curr_cols if c not in core_cols]
      df = df.select(core_cols + generated_cols)
      self.tracker.end_phase()

      #CHECKPOINT 1
      temp_path = self.output_dir / f'{split_name}_checkpoint1.parquet'
      logger.info(f"  Saving Checkpoint 1 -> {temp_path.name}")
      df.sink_parquet(
        temp_path, 
        compression='zstd', 
        statistics=True
        )
      df = pl.scan_parquet(temp_path)
      gc.collect()

      #ROLLING BATCH 1 (Counts) + BATCH 2 (Volumes)
      self.tracker.start_phase(f"{split_name} - Rolling Counts and Volumes")

      logger.info("  Phase 4a: Rolling counts...")
      df = compute_rolling_features_batch1(df)
      df = compute_rolling_features_batch2(df)
      self.tracker.end_phase()

      #CHECKPOINT 2
      temp_path = self.output_dir / f'{split_name}_checkpoint2.parquet'
      logger.info(f"  Saving Checkpoint 2 -> {temp_path.name}")
      df.sink_parquet(temp_path)
      
      #if this is the training split, we compute toxic corridors now
      # using this checkpoint, as requested by the user logic.
      if split_name == 'train' and self.toxic_corridors is None:
          logger.info("   Computing toxic corridors from current state...")
          self.toxic_corridors = self.compute_toxic_corridor_features(df)
          toxic_corridors = self.toxic_corridors
        
      df = pl.scan_parquet(temp_path)
      gc.collect()

      self.tracker.start_phase(f"{split_name} - Rolling Statistics")
      try:
          df = compute_rolling_features_batch3(df)
      except Exception as e:
          logging.error(f" CRITICAL: Rolling stats failed : {e}")
          raise
      self.tracker.end_phase()

    
      #DERIVED FEATURES
      self.tracker.start_phase(f"{split_name} - Derived Features")
      logger.info("  Phase 5a: Derived features...")
      df = compute_derived_features(df)
      self.tracker.end_phase()

      self.tracker.start_phase(f"{split_name} - Advanced Features")
      logger.info(" Phase 5b: Advanced Inflow, Diversity and Sequence Features")
      df = compute_advanced_features(df)
      self.tracker.end_phase()

      #TOXIC CORRIDORS
      if toxic_corridors is not None:
          self.tracker.start_phase(f"{split_name} - Toxic Corridors")
          logger.info("  Phase 6: Toxic corridor features...")
          df = apply_toxic_corridor_features(df, toxic_corridors)
          self.tracker.end_phase()
      
      gc.collect()
      # FINAL OUTPUT
      output_path = self.output_dir / f'{split_name}_features.parquet'
      logger.info(f"  Saving Final Output -> {output_path.name}")
      df.sink_parquet(output_path)

      # CLEANUP CHECKPOINTS
      self._cleanup_checkpoints(split_name)

      logger.info(f"   {split_name.upper()} complete!")



  def _cleanup_checkpoints(self, split_name: str):
        """Safely remove checkpoint files."""
        for i in range(1, 3):
            checkpoint = self.output_dir / f'{split_name}_checkpoint{i}.parquet'
            try:
                if checkpoint.exists():
                    checkpoint.unlink()
                    logger.debug(f" Deleted {checkpoint.name}")
            except Exception as e:
                logger.warning(f"Could not delete {checkpoint}: {e}")

  def run(self):
        """Execute complete pipeline."""
        logger.info("="*60)
        logger.info("AML FEATURE ENGINEERING PIPELINE (OPTIMIZED)")
        logger.info("="*60)
        
        if self.output_dir.exists():
            logger.info(f"Removing existing output directory: {self.output_dir}")
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                logger.error(f" Failed to remove output directory: {e}")
        self.output_dir.mkdir(exist_ok=True)

        #load and Split
        trans, accounts = self.load_data()
        trans = self.create_temporal_split(trans)

        #save Raw Splits (Low memory overhead)
        self.split_and_save_raw(trans)

        #clean up large lazy frame from memory
        del trans, accounts
        import gc
        gc.collect()

        #process all splits 
        for split_name in ['train', 'val', 'test']:
            self.process_split(split_name, toxic_corridors=self.toxic_corridors)
            gc.collect()

        logger.info(' Cleaning up raw intermediate files...')
        for split_name in ['train', 'val', 'test']:
            raw_file = self.output_dir / f"{split_name}_raw.parquet"
            try:
                if raw_file.exists():
                    raw_file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {raw_file}: {e}")

        logger.info("="*60)
        logger.info(" PIPELINE COMPLETE")
        logger.info("="*60)

        summary = self.tracker.get_summary()
        print(summary)

        metric_path = self.output_dir / 'performance_metrics.csv'
        summary.write_csv(metric_path)
        logger.info(f"Performance metrics saved to {metric_path.absolute()}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")



if __name__ == "__main__":

    # Assuming files are in current directory, adjust if needed
    trans_csv = DATA_DIR / 'HI-Medium_Trans.csv'
    acc_csv = DATA_DIR / 'HI-Medium_accounts.csv'

    if not Path(trans_csv).exists():
        print(f"ERROR: {trans_csv} not found. Please download the dataset first.")
    else:
        pipeline = AMLFeatureEngineer(
            transactions_path=trans_csv,
            accounts_path=acc_csv,
            output_dir='./aml_features'
        )
        pipeline.run()