"""
27_rule_baselines.py — Rule-Based Baselines vs ML

Protocols often use simple heuristics to filter sybils before ML is applied.
This experiment evaluates common rule-based approaches and compares them to
our LightGBM model.

Rules:
  R1: wallet_age_days < 30   (new wallet = likely bot)
  R2: tx_count < 10          (low activity = likely bot)
  R3: buy_collections == 1   (only bought one NFT type = likely bot)
  R4: R1 OR R2               (union rule)
  R5: buy_count > 75th pct   (hyperactive buyer — recall-oriented)

Also adds Random Forest and Logistic Regression as ML baselines.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '/Users/adelinewen/Desktop/pre-airdrop-detection/data'
feats_path = f'{OUT_DIR}/nft_feats_labeled_T30.csv'

df = pd.read_csv(feats_path)
FEATURE_COLS = [c for c in df.columns if c not in ['address', 'is_sybil']]
X = df[FEATURE_COLS].values
y = df['is_sybil'].values

bc_75 = df.loc[df['is_sybil']==1, 'buy_count'].quantile(0.75)

# ── Rule-based classifiers (deterministic scores) ────────────────────────────
def evaluate_rule(scores, y, name):
    # AUC needs continuous scores; for binary rules use the 0/1 as score
    auc = roc_auc_score(y, scores)
    preds = (scores >= 0.5).astype(int)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    f1    = f1_score(y, preds, zero_division=0)
    return {'method': name, 'auc': round(auc, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1': round(f1, 4)}

r1 = (df['wallet_age_days'] < 30).astype(float).values
r2 = (df['tx_count'] < 10).astype(float).values
r3 = (df['buy_collections'] == 1).astype(float).values
r4 = np.clip(r1 + r2, 0, 1)
r5 = (df['buy_count'] > bc_75).astype(float).values

rule_results = [
    evaluate_rule(r1, y, 'Rule: wallet_age < 30d'),
    evaluate_rule(r2, y, 'Rule: tx_count < 10'),
    evaluate_rule(r3, y, 'Rule: single NFT type'),
    evaluate_rule(r4, y, 'Rule: age<30 OR tx<10'),
    evaluate_rule(r5, y, 'Rule: high buy_count'),
]

print("Rule-based baselines:")
for r in rule_results:
    print(f"  {r['method']:35s}  AUC={r['auc']:.4f}  F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

# ── ML baselines: RF and LR with 5-fold CV ───────────────────────────────────
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_eval(model, X, y, name, scale=False):
    aucs, f1s, precs, recs = [], [], [], []
    scaler = StandardScaler() if scale else None
    for tr, val in SKF.split(X, y):
        Xtr, Xval = X[tr], X[val]
        if scale:
            Xtr = scaler.fit_transform(Xtr)
            Xval = scaler.transform(Xval)
        model.fit(Xtr, y[tr])
        p = model.predict_proba(Xval)[:, 1]
        preds = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[val], p))
        f1s.append(f1_score(y[val], preds, zero_division=0))
        precs.append(precision_score(y[val], preds, zero_division=0))
        recs.append(recall_score(y[val], preds, zero_division=0))
    return {
        'method': name,
        'auc':       round(np.mean(aucs), 4),
        'precision': round(np.mean(precs), 4),
        'recall':    round(np.mean(recs), 4),
        'f1':        round(np.mean(f1s), 4),
    }

print("\nML baselines (5-fold CV):")
rf_res  = cv_eval(RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                          random_state=42, n_jobs=-1), X, y, 'Random Forest')
lr_res  = cv_eval(LogisticRegression(class_weight='balanced', max_iter=1000,
                                      random_state=42), X, y, 'Logistic Regression', scale=True)
lgb_res = cv_eval(lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                       class_weight='balanced', random_state=42,
                                       n_jobs=-1, verbose=-1), X, y, 'LightGBM (ours)')

for r in [lr_res, rf_res, lgb_res]:
    print(f"  {r['method']:35s}  AUC={r['auc']:.4f}  F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

# ── Combine and save ──────────────────────────────────────────────────────────
all_results = rule_results + [lr_res, rf_res, lgb_res]
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('auc', ascending=False).reset_index(drop=True)
results_df.to_csv(f'{OUT_DIR}/rule_baselines.csv', index=False)
print(f"\n{results_df.to_string(index=False)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#94a3b8'] * 5 + ['#60a5fa', '#f59e0b', '#22c55e']
methods = results_df['method'].tolist()
aucs    = results_df['auc'].tolist()

bars = ax.barh(range(len(methods)), aucs, color=colors[::-1], height=0.6)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods[::-1] if False else results_df['method'].tolist()[::-1], fontsize=10)
ax.set_xlabel('AUC')
ax.set_title('Rule-Based vs ML Baselines (T-30, 5-fold CV)', fontweight='bold')
ax.axvline(0.803, color='#ef4444', linestyle='--', alpha=0.7, label='ARTEMIS 0.803')
ax.set_xlim(0, 1.05)
ax.legend(fontsize=9)

for i, (bar, v) in enumerate(zip(ax.patches, results_df['auc'].tolist()[::-1])):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/rule_baselines.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: rule_baselines.csv + rule_baselines.png")
