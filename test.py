import polars as pl

file = r"E:\AML\dvc_data\processed_with_anomaly\test_features.parquet"

df = pl.scan_parquet(file)
cols = df.collect_schema().names()

fraud = df.filter(pl.col("Is Laundering") == 1).head(1).collect()

if len(fraud) == 0:
    print("no fraud cased")
    exit()

feat_dict = fraud.to_dicts()[0]
from datetime import datetime, date

# dt_cols = [k for k, v in feat_dict.items() if isinstance (v, (datetime, date))]
# print(dt_cols)

cols_to_drop = ["Is Laundering", 
                'Timestamp',
                'Account_HASHED',
                'Account Number_HASHED',
                'Entity ID_HASHED',
                'account_first_txn',
]
for col in cols_to_drop:
   if col in feat_dict:
       del feat_dict[col]

import json
json_out = json.dumps({"features": feat_dict}, indent=2)
print(json_out)