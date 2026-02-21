"""
Experiment 28 — LayerZero Temporal Ablation
============================================
Validates pre-airdrop detection capability on LayerZero (bridge protocol),
mirroring the temporal ablation done on Blur NFT data (Exp 03 / 12).

LZ T0 : 2024-06-20 (LayerZero airdrop date)
Dataset: 19,480 active addresses, 9,899 sybil / 9,581 normal (sybil_rate 0.508)

Baselines for comparison
  Blur   T-30  AUC = 0.9045  (Exp 03 / 12)
  LZ     in-domain (Exp 24)  AUC = 0.892
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# ── paths ─────────────────────────────────────────────────────────────────────
LZ_CSV = "/Users/adelinewen/Desktop/dataset/layerzero/lz_temporal_features.csv"
OUT_JSON = os.path.join(os.path.dirname(__file__), "data", "exp28_lz_temporal_results.json")

# ── constants ─────────────────────────────────────────────────────────────────
CUTOFFS = [30, 60, 90]

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

BLUR_BASELINE_AUC = 0.9045   # Exp 03/12 Blur T-30
LZ_INDOMAIN_AUC   = 0.892    # Exp 24 LayerZero in-domain

# ── step 1 — check data availability ─────────────────────────────────────────
if not os.path.exists(LZ_CSV):
    print("LZ temporal features not ready yet. Run fetch_lz_temporal.mjs first.")
    sys.exit(0)

# ── step 2 — load CSV ─────────────────────────────────────────────────────────
print(f"Loading {LZ_CSV} …")
df = pd.read_csv(LZ_CSV)
df = df.fillna(0)

n_sybil = int(df["is_sybil"].sum())
n_normal = int((df["is_sybil"] == 0).sum())
print(f"  Loaded {len(df):,} rows | sybil={n_sybil:,} "
      f"| normal={n_normal:,} "
      f"| sybil_rate={df['is_sybil'].mean():.3f}")

# Guard: need both classes for classification
if n_sybil == 0 or n_normal == 0:
    print(
        "\n⚠️  DATA ISSUE: lz_temporal_features.csv contains only one class "
        f"(sybil={n_sybil}, normal={n_normal})."
    )
    print(
        "   The fetch script only processed sybil addresses. "
        "Normal (non-sybil) addresses must be added to addresses_labeled.csv "
        "and re-fetched before running this experiment."
    )
    print(
        "   Hint: merge non-sybil addresses from the full LayerZero active-wallet "
        "list into addresses_labeled.csv (is_sybil=0) and re-run fetch_lz_temporal.mjs."
    )
    sys.exit(1)

y_all = df["is_sybil"].values

# ── step 3 — train + evaluate for each cutoff ─────────────────────────────────
results = {}
table_rows = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for T in CUTOFFS:
    tag = f"T{T}"
    feat_cols = [
        f"tx_count_T{T}",
        f"wallet_age_days_T{T}",
        f"unique_contracts_T{T}",
        f"active_span_days_T{T}",
        f"total_volume_T{T}",
    ]

    # 3a — select feature subset
    Xfull = df[feat_cols].values
    yfull = y_all.copy()

    # 3b — drop rows where all 5 features are 0 (no pre-cutoff activity)
    active_mask = ~(Xfull == 0).all(axis=1)
    X = Xfull[active_mask]
    y = yfull[active_mask]
    n_active = int(active_mask.sum())

    print(f"\nCutoff T-{T:02d}: {n_active:,} active addresses "
          f"(dropped {(~active_mask).sum():,} with zero activity)")

    # 3c — 5-fold stratified CV
    clf = LGBMClassifier(**LGBM_PARAMS)

    # Predict probabilities for all samples via cross_val_predict
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    # 3d — mean AUC across folds (replicate fold-by-fold for consistency)
    fold_aucs = []
    for train_idx, val_idx in cv.split(X, y):
        clf_fold = LGBMClassifier(**LGBM_PARAMS)
        clf_fold.fit(X[train_idx], y[train_idx])
        p_val = clf_fold.predict_proba(X[val_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[val_idx], p_val))

    mean_auc = float(np.mean(fold_aucs))

    # 3e — precision / recall / f1 at threshold 0.5
    y_pred = (proba >= 0.5).astype(int)
    prec  = float(precision_score(y, y_pred, zero_division=0))
    rec   = float(recall_score(y, y_pred, zero_division=0))
    f1    = float(f1_score(y, y_pred, zero_division=0))

    results[tag] = {
        "auc":       round(mean_auc, 4),
        "precision": round(prec,     4),
        "recall":    round(rec,      4),
        "f1":        round(f1,       4),
        "n_active":  n_active,
    }
    table_rows.append((f"T-{T}", n_active, mean_auc, prec, rec, f1))

# ── step 4 — print results table ──────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'Cutoff':<8} {'Active Addresses':>18} {'AUC':>7} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 72)
for cutoff, n_active, auc, prec, rec, f1 in table_rows:
    print(f"{cutoff:<8} {n_active:>18,} {auc:>7.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")
print("=" * 72)

# ── step 5 — compare to baselines ─────────────────────────────────────────────
t30_auc = results["T30"]["auc"]
print(f"\nBaseline comparison:")
print(f"  Blur T-30  (Exp 03/12) AUC = {BLUR_BASELINE_AUC:.4f}")
print(f"  LZ in-domain (Exp 24)  AUC = {LZ_INDOMAIN_AUC:.4f}")
print(f"  LZ T-30    (this exp)  AUC = {t30_auc:.4f}  "
      f"({'▲' if t30_auc > LZ_INDOMAIN_AUC else '▼'} vs LZ in-domain, "
      f"{'▲' if t30_auc > BLUR_BASELINE_AUC else '▼'} vs Blur baseline)")

# ── step 6 — save JSON ────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nResults saved → {OUT_JSON}")

# ── step 7 — conclusion ───────────────────────────────────────────────────────
confirms = t30_auc >= 0.85   # conservative threshold for "confirms"
verdict  = "confirms" if confirms else "does not confirm"
print(
    f"\nLayerZero temporal validation: AUC at T-30 = {t30_auc:.3f}. "
    f"This {verdict} that pre-airdrop behavioral signals generalize to bridge protocols."
)
