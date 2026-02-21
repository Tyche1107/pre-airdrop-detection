"""
Phase 1: LightGBM Baseline
===========================
Train on pre-airdrop features, predict Sybil (any_flag = 1).
Temporal ablation: evaluate at different feature windows.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              classification_report, roc_auc_score)
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
df = pd.read_csv(DATA_DIR / "features_baseline.csv")
print(f"Loaded {len(df)} rows, {df['any_flag'].mean():.1%} positive (Sybil)")

# ── Feature sets ────────────────────────────────────────────────────────────
# Features that are STRICTLY pre-airdrop (T < T0)
FEATURES_STRICT = [
    "preseason_external_tx",   # tx count before Season 2 — strictly pre-airdrop
    "is_new_account",          # 1 if preseason_tx == 0
    "funder_wallet_count",     # BW: how many wallets share same funder
    "has_batch_funder",        # BW: binarized (>=5 siblings)
]

# Extended features (includes Season 2 activity — not strictly pre-T0)
FEATURES_EXTENDED = FEATURES_STRICT + [
    "op_count_season2",        # activity during Season 2 (proxy)
    "is_hf_by_80pct",          # HF detector result
]

TARGET = "any_flag"

# ── Train / evaluate ─────────────────────────────────────────────────────────
def evaluate(feature_set, label, target=TARGET):
    X = df[feature_set].fillna(0).values
    y = df[target].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {"precision": [], "recall": [], "f1": [], "auc": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=pos_weight,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        results["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        results["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        results["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        results["auc"].append(roc_auc_score(y_val, y_prob))

    print(f"\n{'='*50}")
    print(f"Feature set: {label}")
    print(f"  Precision : {np.mean(results['precision']):.3f} ± {np.std(results['precision']):.3f}")
    print(f"  Recall    : {np.mean(results['recall']):.3f} ± {np.std(results['recall']):.3f}")
    print(f"  F1        : {np.mean(results['f1']):.3f} ± {np.std(results['f1']):.3f}")
    print(f"  ROC-AUC   : {np.mean(results['auc']):.3f} ± {np.std(results['auc']):.3f}")

    return {k: np.mean(v) for k, v in results.items()}

# ── Per-flag analysis ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Evaluating per-flag targets:")
for flag in ["bw_flag", "ml_flag", "fd_flag", "hf_flag", "any_flag"]:
    pos = df[flag].sum()
    rate = df[flag].mean()
    print(f"  {flag}: {pos} positive ({rate:.1%})")

# ── Run evaluations ────────────────────────────────────────────────────────────
print("\n\nRunning LightGBM with 5-fold CV...")
r1 = evaluate(FEATURES_STRICT,   "Strict pre-airdrop features only")
r2 = evaluate(FEATURES_EXTENDED, "Extended (includes Season 2 activity)")

# ── Feature importance ─────────────────────────────────────────────────────────
print("\n\nTraining final model for feature importance...")
X = df[FEATURES_EXTENDED].fillna(0).values
y = df[TARGET].values
pos_weight = (y == 0).sum() / (y == 1).sum()

final_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05,
    num_leaves=31, scale_pos_weight=pos_weight,
    random_state=42, verbose=-1
)
final_model.fit(X, y)

importance = sorted(zip(FEATURES_EXTENDED, final_model.feature_importances_),
                    key=lambda x: x[1], reverse=True)
print("\nFeature Importance (split):")
for feat, imp in importance:
    bar = "█" * int(imp / max(v for _, v in importance) * 30)
    print(f"  {feat:<30} {bar} ({imp})")

print("\n✅ Done. Next step: 03_temporal_ablation.py")
