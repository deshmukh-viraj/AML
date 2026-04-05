# Building a Trustworthy AML Detection Model
### What the Terminal Taught Us About Evaluation, Calibration, and Data Integrity

---

## Background

We built a supervised machine learning pipeline to detect money laundering on the IBM
HI-Medium synthetic transaction dataset  roughly 25 million transactions, 23,844 confirmed
fraud cases, and a severe class imbalance of 1 legitimate transaction for every 104 fraudulent
ones.

The goal was straightforward: catch as much fraud as possible while keeping false alarms
under 1% of all legitimate transactions. We evaluated four models  XGBoost, LightGBM,
RandomForest, and LogisticRegression  and selected the most reliable one using a technique
called Monte Carlo Cross Validation (MCCV).

What followed was a series of warnings, investigations, and fixes that taught us more about
building trustworthy models than any tutorial ever could.

---

## Issue 1  The Model That Looked Too Good

### What We Saw In The Terminal

After our first full run, MCCV reported excellent results:

```
LightGBM MCCV recall: 0.8531 ± 0.0037  (conservative: 0.8458)
```

85% recall  meaning the model would catch 85 out of every 100 fraud cases while keeping
false alarms under 1%. We were pleased.

Then the next line printed:

```
WARNING — LightGBM: MCCV recall 0.8531 vs val recall 0.4805 — gap=0.3725
```

The model that claimed 85% recall was actually delivering 48% on real validation data.
A gap of 0.37. One of these numbers was lying.

---

### Why This Happened

Think of it like a student who practises on last year's exam papers and scores 95%. But when
the actual exam arrives with new questions, they score 60%. The practice was too easy because
it used the same material.

Our MCCV was doing exactly this. It was randomly shuffling the training data and testing
itself on random slices  meaning it always tested on data from the same time period it
trained on. In the real world, a fraud model always predicts the future. Fraudsters from
January behave slightly differently in February. Our evaluation never accounted for this.

```
What MCCV was doing (wrong):
Train on random 80% of days 1-11 → Test on random 20% of days 1-11
→ Same time window → Artificially high scores

What actually happens in production:
Train on days 1-11 → Predict on days 12-13
→ Future data → Real performance
```

There is also a regulatory reason this matters. Under banking model risk frameworks, a
model's evaluation must reflect how it will actually be used. Presenting an 85% recall
number to a compliance team when the real forward-looking performance is 48% would fail
any independent model review.

---

### How We Fixed It — Walk-Forward Validation

We replaced the random shuffling with an **expanding window walk-forward** approach — the
industry standard for time-series financial data.

Instead of randomly picking training and test slices, each fold now trains on everything
available up to a certain point in time and tests on the immediately following window:

```
Fold 0: Train on first 50% of data  → Test on next 10%
Fold 1: Train on first 60% of data  → Test on next 10%
Fold 2: Train on first 70% of data  → Test on next 10%
Fold 3: Train on first 80% of data  → Test on next 10%
Fold 4: Train on first 90% of data  → Test on next 10%
```

Each fold simulates a real deployment  train on all known history, predict what comes next.
The MCCV score we get from this is lower but honest. And it now aligns with what we actually
see on validation data because both respect the same rule: the past trains the model, the
future tests it.

**Result:** Gap reduced from **0.37 -> 0.24**, a 43% improvement. The remaining gap is
expected  we are validating on only 2 days of future data after training on 11 days. Some
gap is healthy. A gap of zero would actually be suspicious.

---

## Issue 2  Probability Scores That Meant Nothing

### What We Saw In The Terminal

```
WARNING — Calibration error: 0.3990 (< 0.05 good, > 0.10 concerning)
WARNING — Probability scores may be misleading.
          Consider Platt scaling before surfacing scores to analysts.
```

The model's probability scores were unreliable. When it said "this transaction has 80%
chance of being fraud", the real rate was nowhere near 80%.

---

### Why This Happened

Imagine a weather forecaster who says "90% chance of rain" but it only rains 30% of those
times. They might still correctly predict which days are rainy versus sunny — they just
cannot be trusted on the actual percentages.

Our model had the same problem for three reasons:

**The class imbalance.** With 1 fraud per 104 legitimate transactions, the model internally
inflates fraud scores to compensate. This overcorrects and makes the raw numbers unreliable.

**Tree models are not natural probability estimators.** XGBoost and LightGBM are excellent
at ranking transactions from most suspicious to least suspicious  but the scores they output
are not true probabilities. They are good for comparison but not for absolute interpretation.

**Temporal shift.** The model was trained on days 1-11 but calibration was measured on days
12-13. A score of 0.6 on day 8 carries a different meaning on day 13 as fraud patterns
evolve slightly.

In AML this matters practically. Compliance analysts act on these scores every day. If a
score of 0.75 triggers an investigation and that threshold is set based on unreliable
probabilities, we are either flooding analysts with false alarms or missing real cases.

---

### How We Fixed It  Platt Scaling

We added a technique called **Platt scaling**  a simple correction layer fitted on top of
the model's raw scores. Think of it as a translator that converts the model's internal
language into true probabilities.

We were careful to keep this honest. Our validation data was already split into two halves:

```
Calibration slice -> used to TEACH the Platt scaler what the scores really mean
Evaluation slice  -> used to MEASURE whether calibration actually worked
```

The scaler never saw the evaluation slice during fitting  same principle as keeping test
data separate from training data.

```python
# Fit on calibration slice only
platt = LogisticRegression()
platt.fit(raw_scores_from_cal_slice, true_labels_cal_slice)

# Apply to evaluation and test
calibrated_score = platt.predict_proba(raw_score)[1]
```

**Result:** Calibration error dropped from **0.3990 -> 0.0011**. When the model now says
80% fraud probability, the true rate is close to 80%. Analysts can trust the numbers.

---

## Issue 3  The Day the Model Learned the Wrong Thing

### What We Saw In SHAP

After running SHAP explainability  a technique that tells us which features the model
relies on most  we found this:

```
#1 feature: day_of_month_sin (importance: 2.2)
#2 feature: fraud_rate (importance: 1.2)
#3 feature: txns_in_directed_pair
```

The most important feature was the day of the month. A fraud detection model should be
learning behavioural patterns  burst transactions, counterparty switching, unusual amounts.
Not calendar dates.

---

### Why This Happened

We ran a quick check on the raw data:

```
Days 1-16  → fraud rate: 0.02% - 0.25%  (normal)
Days 17-28 → fraud rate: 57% - 63%  (almost pure fraud)
```

The IBM HI-Medium dataset is synthetic  fraud was artificially injected into days 17-28
during data generation. This created a cliff: day 16 has 0.08% fraud, day 17 has 59% fraud.

The model learned this cliff perfectly. It did not learn "this account sent 15 transactions
to 8 different people in one hour"  it learned "it is day 17, therefore fraud." That is
not a typology. That is a calendar lookup.

In a real production deployment, there would be no such cliff. Day 17 in real banking data
looks like day 16. A model that relies on this would generate a flood of false alerts on
the 17th of every month for no reason  and miss real fraud that happens on the 3rd.

---

### How We Fixed It

We removed `day_of_month_sin` and `day_of_month_cos` from the feature engineering pipeline
in `base_features.py`. Importantly, we did not remove the fraud transactions from days 17-28
from training data  those cases are still there with all their behavioural features intact.

What we removed was the model's ability to use the calendar date as a shortcut. By taking
away the easy answer, the model was forced to learn the real patterns.

```
Before removal: Model says "day 17 → fraud"
After removal:  Model says "15 transactions in 1 hour to 8 counterparties → fraud"
```

**Result:** AUC improved from **0.850 -> 0.883**, and the top SHAP features shifted to
genuine AML typologies  burst velocity, directed pair transactions, counterparty switching,
account tenure. Removing the shortcut made the model stronger and more generalisable.

---

## Issue 4  Features That Had No History

### What We Found

Rolling features like burst score, counterparty entropy, and velocity are computed per
account over time. They need historical context  an account's behaviour over the past 28
days  to be meaningful.

When we processed validation and test data in isolation, those features had no memory of
the training period:

```
Train accounts -> 11 days of transaction history -> rich rolling features 
Val accounts   -> only see 2 days of data        -> features are near-empty 
```

A suspicious account that had been sending 50 rapid transactions per day for 10 days would
look completely clean on day 12  because the validation features had forgotten everything
that happened before.

---

### How We Fixed It

Before computing rolling features for validation and test batches, we prepended the training
history for each account. The features were computed on the full timeline, then the training
rows were dropped  leaving validation rows with properly populated historical features.

```
Process val batch:
1. Add training rows for same accounts (as history context)
2. Sort by account + timestamp
3. Compute all 8 feature steps (rolling, entropy, network, etc.)
4. Drop the training rows
5. Save only the val rows  now with full historical context
```

This same fix applied automatically to all time-dependent features  rolling velocity,
counterparty entropy, network features  because they all run within the same pipeline.

---

## Issue 5  Dead Features and Misaligned Columns

### Two Problems Found During Code Review

**Dead features:** Some features had nearzero variance across 1.5 million rows  meaning
they barely changed between transactions. A feature that is almost always the same value
cannot help a model distinguish fraud from legitimate activity. We added a variance filter
to remove these before training.

**Column misalignment:** When we applied the variance filter to training data, we made an
error in how we applied the same filter to validation data:

```python
# Wrong — takes the first N columns regardless of which were removed
X_val = X_val[:, :len(features)]

# Correct — applies the exact same column mask
X_val = X_val[:, variance_mask]
```

This bug caused training and validation data to have completely different features in each
column position. The model was trained on one set of features but evaluated on a scrambled
version. This produced a catastrophic gap of 0.78 in one run  far worse than anything we
had seen before.

Once identified and fixed, the gap returned to its expected level.

---

## Final Results

After all fixes applied:

| Metric | Value |
|---|---|
| AUC | **0.943** |
| MCCV vs Val gap | **0.24** |
| Calibration error | **0.0011** |
| Top SHAP feature | anomaly_score (genuine signal) |
| Operational threshold | saved with model for deployment |

The journey from our first run to this result involved fixing evaluation methodology,
calibration, data leakage, feature history, and a column alignment bug. Each fix made
the model more honest rather than more impressive on paper.

---

## The Lesson

> **A model that looks good is not the same as a model that is good.**

Every number we improved in this pipeline came from making the evaluation more honest, not
from making the model more complex. The final AUC of 0.943 is trustworthy precisely because
we spent more time questioning our methodology than tuning our hyperparameters.

In AML  where the output feeds into regulatory decisions, analyst workloads, and potential
criminal investigations  a model you can explain and defend is worth more than one with
impressive numbers you cannot fully account for.

---

*Built on IBM HI-Medium synthetic AML dataset. Pipeline: Polars -> temporal feature
engineering with train history prepend > Isolation Forest anomaly scores → XGBoost/LightGBM/
RF/LR with expanding window MCCV > Platt calibration > SHAP explainability >
MLflow tracking via DagsHub.*