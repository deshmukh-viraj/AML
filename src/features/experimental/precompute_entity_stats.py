import polars as pl
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def precompute_entity_stats(accounts: pl.DataFrame) -> pl.DataFrame:
    """
    Compute entity statistics once, reuse for all splits.
    OPTIMIZATION: Prevents 3x redundant computation.
    """
    logger.info("🔧 Pre-computing entity statistics...")
    
    entity_stats = (
        accounts.lazy()
        .group_by('Entity ID')
        .agg([
            pl.count().alias('entity_account_count'),
            pl.col('Bank ID').n_unique().alias('entity_bank_count'),
        ])
        .collect()
    )
    
    # Join back to accounts for easier merging
    accounts_enriched = accounts.join(
        entity_stats,
        on='Entity ID',
        how='left'
    ).select([
        'Account Number',
        'Entity ID', 
        'entity_account_count',
        'entity_bank_count'
    ])
    
    logger.info(f" Entity stats computed for {len(accounts_enriched)} accounts")
    return accounts_enriched