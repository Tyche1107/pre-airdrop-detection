"""
Threshold Analysis + Precision-Recall Curve
=============================================
The strict pre-airdrop model has AUC=0.715 but F1=0 at default threshold.
Find the optimal threshold using precision-recall curve.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_curve, f1_score,
                              roc_auc_score, average_precision_score)
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR  = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "features_baseline.csv")
print(f"Loaded {len(df)} rows, {df['any_flag'].mean():.1%} Sybil")

FEATURES_STRICT = [
    "preseason_external_tx",
    "is_new_account",
    "funder_wallet_count",
    "has_batch_funder",
]

FEATURES_EXTENDED = FEATURES_STRICT + [
    "op_count_season2",
    "is_hf_by_80pct",
]

TARGET = "any_flag"

def train_and_get_probs(features):
    X = df[features].fillna(0).values
    y = df[TARGET].values
    pos_weight = (y == 0).sum() / (y == 1).sum()

    all_probs = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=31, scale_pos_weight=pos_weight,
            random_state=42, verbose=-1
        )
        model.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        all_probs[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    return y, all_probs

print("\nComputing probability scores (5-fold)...")

# ── Strict pre-airdrop features ───────────────────────────────────────────
y, probs_strict = train_and_get_probs(FEATURES_STRICT)
precisions_s, recalls_s, thresholds_s = precision_recall_curve(y, probs_strict)
f1_scores_s = 2 * precisions_s * recalls_s / (precisions_s + recalls_s + 1e-8)
best_idx_s = np.argmax(f1_scores_s)
best_thresh_s = thresholds_s[best_idx_s] if best_idx_s < len(thresholds_s) else 0.5

auc_s = roc_auc_score(y, probs_strict)
ap_s  = average_precision_score(y, probs_strict)

print(f"\n[Strict pre-airdrop features]")
print(f"  ROC-AUC:        {auc_s:.3f}")
print(f"  Avg Precision:  {ap_s:.3f}")
print(f"  Best threshold: {best_thresh_s:.3f}")
print(f"  Best F1:        {f1_scores_s[best_idx_s]:.3f}")
print(f"  Precision:      {precisions_s[best_idx_s]:.3f}")
print(f"  Recall:         {recalls_s[best_idx_s]:.3f}")

# ── Extended features ─────────────────────────────────────────────────────
y, probs_ext = train_and_get_probs(FEATURES_EXTENDED)
precisions_e, recalls_e, thresholds_e = precision_recall_curve(y, probs_ext)
f1_scores_e = 2 * precisions_e * recalls_e / (precisions_e + recalls_e + 1e-8)
best_idx_e = np.argmax(f1_scores_e)
best_thresh_e = thresholds_e[best_idx_e] if best_idx_e < len(thresholds_e) else 0.5

auc_e = roc_auc_score(y, probs_ext)
ap_e  = average_precision_score(y, probs_ext)

print(f"\n[Extended features (includes Season 2)]")
print(f"  ROC-AUC:        {auc_e:.3f}")
print(f"  Avg Precision:  {ap_e:.3f}")
print(f"  Best threshold: {best_thresh_e:.3f}")
print(f"  Best F1:        {f1_scores_e[best_idx_e]:.3f}")
print(f"  Precision:      {precisions_e[best_idx_e]:.3f}")
print(f"  Recall:         {recalls_e[best_idx_e]:.3f}")

# ── Plot: Precision-Recall curves ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PR Curve
ax = axes[0]
ax.plot(recalls_s, precisions_s, 'b-', linewidth=2,
        label=f'Strict pre-airdrop (AP={ap_s:.3f})')
ax.plot(recalls_e, precisions_e, 'r-', linewidth=2,
        label=f'Extended (AP={ap_e:.3f})')
ax.axhline(y=df[TARGET].mean(), color='gray', linestyle='--', alpha=0.5,
           label=f'Baseline (random): {df[TARGET].mean():.3f}')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# F1 vs Threshold
ax = axes[1]
ax.plot(thresholds_s, f1_scores_s[:-1], 'b-', linewidth=2,
        label=f'Strict (best F1={f1_scores_s[best_idx_s]:.3f} @ {best_thresh_s:.2f})')
ax.plot(thresholds_e, f1_scores_e[:-1], 'r-', linewidth=2,
        label=f'Extended (best F1={f1_scores_e[best_idx_e]:.3f} @ {best_thresh_e:.2f})')
ax.axvline(x=best_thresh_s, color='blue', linestyle=':', alpha=0.7)
ax.axvline(x=best_thresh_e, color='red', linestyle=':', alpha=0.7)
ax.set_xlabel('Decision Threshold', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score vs. Threshold', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "precision_recall_threshold.png", dpi=150, bbox_inches='tight')
print(f"\nFigure saved: figures/precision_recall_threshold.png")
print("\n✅ Done. Results:")
print(f"  Strict pre-airdrop:  F1={f1_scores_s[best_idx_s]:.3f}  AUC={auc_s:.3f}")
print(f"  Extended:            F1={f1_scores_e[best_idx_e]:.3f}  AUC={auc_e:.3f}")
