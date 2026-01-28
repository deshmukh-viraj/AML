import polars as pl

def add_benford_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute Benford's Law deviation and round number analysis uisng math.
    """
    return df.with_columns([
        #extract first digit of amount
        pl.col('Amount Paid').abs()
         .log10()
         .mod(1)
         .mul(10)
         .floor()
         .cast(pl.Int32, strict=False)
         .alias('first_digit'),
       
        #check if amount is round number (divisible by 100)
        (pl.col('Amount Paid') % 100 == 0).cast(pl.Int8).alias('is_round_100'),

        #check if amount is very round (divisible by 1000)
        (pl.col('Amount Paid') % 1000 == 0).cast(pl.Int8).alias('is_round_1000'),
    ])
