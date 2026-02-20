# Simple AML Lab - Documentation

## Overview

Simple AML Lab is a lightweight experimentation framework designed for Anti-Money Laundering (AML) model exploration. It provides a clean, modular pipeline for comparing balancing strategies and classification models on highly imbalanced financial transaction datasets.

## Folder Structure

```
simple-aml-lab/
├── input/                     # Input data (gitignored)
│   └── transactions.parquet   # Your feature-engineered dataset
├── outputs/                   # Experiment results
│   └── experiment_results.csv # Aggregated results log
├── src/                       # Core modules
│   ├── __init__.py
│   ├── balancing.py           # Class imbalance handling
│   └── modeling.py            # Model definitions & metrics
├── config.py                  # Hyperparameter configuration
├── orchestrator.py            # Main execution script
└── README.md                  # This file
```

## Quick Start

### 1. Prepare Your Data

Place your parquet file in the `input/` directory. Expected format:

```
DataFrame with:
- Target column: is_laundering (0/1)
- Feature columns: numeric features
```

### 2. Run Experiments

```bash
python orchestrator.py --data_path input/transactions.parquet
```

### 3. View Results

Results are saved to `outputs/experiment_results.csv`.

---

## Design Decisions for AML

### Phase 1: Balancing Strategies

#### Why These Methods?

| Method | Rationale for AML |
|--------|-------------------|
| **none (baseline)** | Establishes baseline; many AML models actually work better without explicit balancing when using tree-based methods |
| **class_weight** | Cost-sensitive learning; doesn't alter data distribution, maintains original patterns |
| **under_sample** | Efficient for large datasets; reduces memory and training time significantly |
| **smote** | Included with warnings; creates synthetic transactions that may not reflect real laundering patterns |

#### Memory-Aware Design

- **Index-based slicing**: Uses pandas index selection rather than full DataFrame copies
- **No in-place mutations**: All functions return new objects, preserving original data
- **Explicit garbage collection**: `gc.collect()` called after heavy operations

#### Why Not SMOTE for Production?

SMOTE generates synthetic fraud patterns that:
- May hallucinate unrealistic transaction behaviors
- Cannot be traced back to real alerts for investigation
- Creates "evidence" that never existed in production systems

### Phase 2: Model Selection

#### Why These Models?

| Model | AML Use Case |
|-------|--------------|
| **Logistic Regression** | Interpretable; useful for risk scoring and explainability requirements |
| **Random Forest** | Robust to outliers; handles mixed feature types; built-in class weighting |
| **LightGBM** | Fast training on large data; excellent for high-dimensional feature spaces |

#### Why Not Neural Networks?

- Require more data for stable training
- Harder to explain to compliance teams
- Overfitting risk with imbalanced data

### Phase 3: Metrics

#### Primary: PR-AUC (Precision-Recall AUC)

**Why PR-AUC over ROC-AUC for AML?**

```
ROC-AUC can show 0.99+ even when model captures ZERO fraud cases.
This happens because TN (legitimate transactions) dominates the matrix.
```

PR-AUC focuses exclusively on the positive class performance:
- Only considers precision and recall
- Unaffected by true negatives
- Reflects actual detection capability

#### Operational Metrics

| Metric | Business Meaning |
|--------|------------------|
| **Recall at 80% Precision** | "If we only investigate alerts with 80% confidence, what % of fraud do we catch?" |
| **Recall at 95% Precision** | "What's our catch rate with high-confidence alerts only?" |

These directly map to operational constraints:
- Investigation team capacity
- False positive costs (customer friction, investigation time)
- Regulatory requirements for alert disposition

---

## Production Promotion Notes

### What to Promote

1. **Balancing Strategy**
   - `class_weight` or `under_sample` are production-safe
   - Document the chosen ratio and rationale

2. **Model Configuration**
   - Tree-based models (RF, LightGBM) are preferred
   - Serialize best model with `joblib` or `pickle`

3. **Feature Pipeline**
   - Move feature engineering to a Feature Store
   - Ensure point-in-time correctness for training

### What Requires Work

1. **Latency Requirements**
   - Tree models may need compilation (Treelite) or ONNX conversion
   - Target <50ms for real-time scoring

2. **Monitoring**
   - Implement drift detection on input features
   - Monitor prediction distribution, not just accuracy

3. **Explainability**
   - Add SHAP or LIME for individual predictions
   - Compliance teams often require reason codes

4. **Governance**
   - Model versioning (not MLflow, consider DVC or custom)
   - Audit trail for hyperparameter changes
   - Documentation of fairness considerations

### Recommended Production Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Feature Store  │────▶│  Model Serving  │────▶│  Alert Queue    │
│  ( Feast )      │     │  ( ONNX/REST ) │     │  ( Investigation)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Drift Monitor  │     │  Metrics Store │
│  ( Evidently )  │     │  ( Prometheus ) │
└─────────────────┘     └─────────────────┘
```

---

## Extending the Framework

### Adding New Models

Edit `src/modeling.py`:

```python
def create_xgboost(**kwargs):
    # Your model definition
    return xgb.XGBClassifier(...)

# Add to registry
MODEL_REGISTRY["xgboost"] = create_xgboost
```

### Adding New Metrics

Edit `compute_aml_metrics()` in `src/modeling.py`:

```python
def compute_aml_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
    # Existing metrics...
    
    # Add new metric
    if y_proba is not None:
        metrics["average_precision"] = average_precision_score(y_true, y_proba)
    
    return metrics
```

---

## Limitations

- **No time-series handling**: Uses random split; for production AML, implement time-based split
- **Single-table assumption**: Doesn't handle relational data directly
- **No feature selection**: Assumes pre-engineered features
- **Limited explainability**: No built-in SHAP/LIME integration

---

## License

MIT License - Free for commercial and academic use.
