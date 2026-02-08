"""
Network Graph Features for AML Detection

Extract network-level patterns from transaction flows:
- Node centrality (degree, betweenness, closeness)
- Clustering coefficients
- PageRank scores
- Community patterns
"""

import polars as pl
import networkx as nx
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def build_transaction_networks(df: pl.LazyFrame) -> Tuple[nx.DiGraph, nx.DiGraph]:
    """
    Build two directed networks from transaction data:
    1. Account-to-Account network (via shared bank relationships)
    2. Bank-to-Bank network (direct transaction corridors)
    
    Returns:
        Tuple of (account_network, bank_network)
    """
    # Collect data needed for network construction
    network_data = df.select([
        'Account_HASHED',
        'From Bank',
        'To Bank',
        'Amount Paid',
        'Amount Received'
    ]).collect()
    
    # Build bank-to-bank network
    bank_network = nx.DiGraph()
    
    # Build account network (accounts connected through shared banks)
    account_network = nx.DiGraph()
    
    for row in network_data.iter_rows(named=True):
        from_bank = row['From Bank']
        to_bank = row['To Bank']
        account = row['Account_HASHED']
        amount_paid = row['Amount Paid']
        
        # Bank network: From Bank -> To Bank
        if bank_network.has_edge(from_bank, to_bank):
            bank_network[from_bank][to_bank]['weight'] += amount_paid
            bank_network[from_bank][to_bank]['count'] += 1
        else:
            bank_network.add_edge(from_bank, to_bank, weight=amount_paid, count=1)
        
        # Account network: this only uses Bank edges for now
        # (could be extended to co-account relationships)
        if not account_network.has_node(account):
            account_network.add_node(account)
    
    return account_network, bank_network


def compute_bank_centrality_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate bank-level centrality and join back to transactions.
    REFACTORED: Pre-aggregate before NetworkX to reduce memory 100x.
    """
    logger.info("Computing bank centrality features...")
    
    # PRE-AGGREGATE: Reduce 32M rows → ~10K bank pairs
    bank_edges = (
        df.select(['From Bank', 'To Bank'])
        .group_by(['From Bank', 'To Bank'])
        .agg(pl.count().alias('weight'))
        .collect(streaming=True)  # Now only ~10K rows
    )
    
    # Build network from aggregated edges (10K iterations vs 32M)
    bank_network = nx.DiGraph()
    for row in bank_edges.iter_rows(named=True):
        from_bank = row['From Bank']
        to_bank = row['To Bank']
        weight = row['weight']
        
        if bank_network.has_edge(from_bank, to_bank):
            bank_network[from_bank][to_bank]['weight'] += weight
        else:
            bank_network.add_edge(from_bank, to_bank, weight=weight)
    
    # Calculate centrality metrics (unchanged)
    out_degree = dict(bank_network.out_degree())
    in_degree = dict(bank_network.in_degree())
    
    try:
        pagerank = nx.pagerank(bank_network, weight='weight')
    except:
        pagerank = {node: 0.0 for node in bank_network.nodes()}
        logger.warning("PageRank calculation failed, using zeros")
    
    try:
        betweenness = nx.betweenness_centrality(bank_network, weight='weight')
    except:
        betweenness = {node: 0.0 for node in bank_network.nodes()}
        logger.warning("Betweenness calculation failed, using zeros")
    
    # Convert to Polars dataframes for efficient joining
    from_bank_features = pl.DataFrame({
        'From Bank': list(out_degree.keys()),
        'from_bank_out_degree': list(out_degree.values()),
        'pagerank_from_bank': [pagerank.get(b, 0.0) for b in out_degree.keys()],
        'betweenness_from_bank': [betweenness.get(b, 0.0) for b in out_degree.keys()]
    })
    
    to_bank_features = pl.DataFrame({
        'To Bank': list(in_degree.keys()),
        'to_bank_in_degree': list(in_degree.values()),
        'pagerank_to_bank': [pagerank.get(b, 0.0) for b in in_degree.keys()],
        'betweenness_to_bank': [betweenness.get(b, 0.0) for b in in_degree.keys()]
    })
    
    # Join back to original dataframe
    df = df.join(from_bank_features.lazy(), on='From Bank', how='left')
    df = df.join(to_bank_features.lazy(), on='To Bank', how='left')
    
    # Fill nulls with 0
    df = df.with_columns([
        pl.col('from_bank_out_degree').fill_null(0),
        pl.col('to_bank_in_degree').fill_null(0),
        pl.col('pagerank_from_bank').fill_null(0.0),
        pl.col('pagerank_to_bank').fill_null(0.0),
        pl.col('betweenness_from_bank').fill_null(0.0),
        pl.col('betweenness_to_bank').fill_null(0.0)
    ])
    
    return df

def compute_account_network_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate account-level network features within rolling windows.
    
    Features:
    - account_counterparty_diversity_7d: Distinct 'To Bank' values in 7d window
    - account_counterparty_diversity_28d: Distinct 'To Bank' values in 28d window
    - account_bank_repeat_rate_7d: % of transactions to repeat banks in 7d window
    """
    logger.info("Computing account-level network features...")
    
    # Ensure data is sorted for time-based rolling operations
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # Counterparty diversity: count distinct 'To Bank' per account per window
    df = df.with_columns([
        pl.col('To Bank')
            .n_unique()
            .rolling_max_by(by='Timestamp', window_size='7d')
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0)
            .cast(pl.UInt32)
            .alias('account_counterparty_diversity_7d'),
        
        pl.col('To Bank')
            .n_unique()
            .rolling_max_by(by='Timestamp', window_size='28d')
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0)
            .cast(pl.UInt32)
            .alias('account_counterparty_diversity_28d'),
    ])
    
    # Bank repeat rate: proportion of repeat banks (using 200-row window ≈ 7d)
    df = df.with_columns([
        # Count occurrences of current bank in recent history
        (pl.col('To Bank').shift(1)
            .over('Account_HASHED')
            .eq(pl.col('To Bank'))
            .cast(pl.Int8)
            .rolling_sum(window_size=200)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0) /
         (pl.len()
             .rolling_sum(window_size=200)
             .over('Account_HASHED')
             .shift(1)
             .fill_null(1)))
        .clip(0.0, 1.0)
        .alias('account_bank_repeat_rate_7d')
    ])
    
    return df


def compute_corridor_risk_score(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Create corridor-level risk aggregation feature.
    
    Combines From Bank -> To Bank corridor information with transaction specifics.
    """
    logger.info("Computing corridor-level features...")
    
    df = df.with_columns([
        # Create a corridor identifier
        (pl.col('From Bank').cast(pl.Utf8) + '_to_' + pl.col('To Bank').cast(pl.Utf8))
            .alias('corridor')
    ])
    
    # Compute corridor statistics in rolling windows (500-row window ≈ 28d)
    df = df.with_columns([
        pl.col('Amount Paid')
            .rolling_mean(window_size=500)
            .over('corridor')
            .shift(1)
            .fill_null(0)
            .alias('corridor_mean_amount_28d'),
        
        pl.col('Amount Paid')
            .rolling_std(window_size=500)
            .over('corridor')
            .shift(1)
            .fill_null(0)
            .alias('corridor_std_amount_28d'),
    ])
    
    return df


def add_network_features(df: pl.DataFrame) -> pl.DataFrame:
    """ 
    Add all network features
    """
    # Defensive check: ensure the input is a Polars DataFrame or LazyFrame to
    # catch regressions where a different type (e.g., a tuple) is passed in.
    assert isinstance(df, (pl.DataFrame, pl.LazyFrame)), (
        "add_network_features expects a Polars DataFrame or LazyFrame"
    )

    logger.info("  Building transaction network...")
    # Build networks separately; do NOT overwrite `df` which should remain the
    # transaction DataFrame. Overwriting `df` with the tuple caused subsequent
    # feature computations to fail (it became a tuple instead of a DataFrame).
    account_network, bank_network = build_transaction_networks(df)

    logger.info("  Computing bank centrality...")
    df = compute_bank_centrality_features(df)

    logger.info("  Computing account network....")
    df = compute_account_network_features(df)

    logger.info("  Computing corridor risk score...")
    df = compute_corridor_risk_score(df)

    # Return the enhanced DataFrame. Keep networks in local variables in case
    # they are needed later; if needed, consider returning them as well.
    return df