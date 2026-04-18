import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# load saved data
y_true = np.load("models\\trained_model\\artifacts\\y_true.npy")
y_prob = np.load("models\\trained_model\\artifacts\\y_prob.npy")

# try thresholds
thresholds = np.linspace(0.01, 0.2, 50)

results = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)

    results.append({
        "threshold": t,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fn": fn,
        "fp": fp
    })

df = pd.DataFrame(results)

# 🔥 Your current method (FPR constraint)
fpr_limit = 0.01
fpr_based = df[df["fpr"] <= fpr_limit].sort_values("recall", ascending=False).head(1)

# 🔥 F1 best
f1_best = df.sort_values("f1", ascending=False).head(1)

# 🔥 High recall option
high_recall = df[df["recall"] >= 0.6].sort_values("precision", ascending=False).head(1)

print("\n=== FPR Constraint Method ===")
print(fpr_based)

print("\n=== Best F1 ===")
print(f1_best)

print("\n=== High Recall Option ===")
print(high_recall)