"""
05_pr_analysis.py — Precision-Recall Analysis & Optimal Threshold

Using the best pre-airdrop model (T-30, AUC=0.904), find optimal threshold
for deployment: max F1, precision@90%, recall@90%.

Also: compare this work vs baseline ARTEMIS (AUC=0.803 from paper).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset")
TXS_PATH = DATA_DIR / "TXS2_1662_1861.csv"
FEATURES_PATH = DATA_DIR / "addresses_all_with_loaylty_blend_blur_label319.csv"
OUT_DIR = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

T0 = 1700525735

# Test with T-30 (most practical: one month before airdrop)
CUTOFF_T30 = T0 - 30 * 86400

def ts_label(ts_unix):
    return datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d")

print("=== Precision-Recall Analysis (T-30 Model) ===\n")

# Load data
_flags = pd.read_csv(DATA_DIR / "airdrop_targets_behavior_flags.csv")
targets = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
del _flags

print("Loading TXS2...")
txs = pd.read_csv(TXS_PATH, usecols=[
    'from', 'send', 'receive', 'timestamp', 'trade_price', 'event_type', 'contract_address'
], dtype={'timestamp': 'Int64', 'trade_price': 'float32'}, low_memory=False)
txs['ts'] = txs['timestamp'] // 1000
txs['from'] = txs['from'].str.lower().fillna('')
txs['send'] = txs['send'].str.lower().fillna('')
txs['receive'] = txs['receive'].str.lower().fillna('')

ext = pd.read_csv(FEATURES_PATH)
ext['address'] = ext['address'].str.lower()
ext_cols = ['address', 'blend_in_count', 'blend_out_count', 'blend_net_value',
            'LP_count', 'DeLP_count', 'unique_interactions', 'ratio']
ext = ext[ext_cols].rename(columns={'address': 'addr'})

def compute_features(txs_sub, cutoff_ts):
    df = txs_sub.copy()
    buys = df[df['event_type'] == 'Sale'][['send', 'contract_address', 'trade_price', 'ts']].rename(columns={'send': 'addr'})
    sells = df[df['event_type'] == 'Sale'][['receive', 'contract_address', 'trade_price', 'ts']].rename(columns={'receive': 'addr'})
    all_addrs = pd.concat([buys[['addr']], sells[['addr']], df[['from']].rename(columns={'from': 'addr'})]).drop_duplicates()
    all_addrs = all_addrs[all_addrs['addr'].str.startswith('0x')]

    buy_stats = buys.groupby('addr').agg(buy_count=('addr','count'), buy_value=('trade_price','sum'), buy_collections=('contract_address','nunique'), buy_last_ts=('ts','max'), buy_first_ts=('ts','min')).reset_index()
    sell_stats = sells.groupby('addr').agg(sell_count=('addr','count'), sell_value=('trade_price','sum')).reset_index()
    from_stats = df.groupby('from').agg(tx_count=('ts','count'), first_tx_ts=('ts','min')).reset_index().rename(columns={'from':'addr'})

    feat = all_addrs.merge(buy_stats, on='addr', how='left').merge(sell_stats, on='addr', how='left').merge(from_stats, on='addr', how='left').fillna(0)
    feat['total_trade_count'] = feat['buy_count'] + feat['sell_count']
    feat['sell_ratio'] = feat['sell_count'] / (feat['total_trade_count'] + 1e-6)
    feat['pnl_proxy'] = feat['sell_value'] - feat['buy_value']
    feat['wallet_age_days'] = (cutoff_ts - feat['first_tx_ts'].clip(lower=0)) / 86400
    feat['days_since_last_buy'] = (cutoff_ts - feat['buy_last_ts'].clip(lower=0)) / 86400
    feat['recent_activity'] = (feat['buy_last_ts'] > (cutoff_ts - 30*86400)).astype(int)
    feat = feat[feat['total_trade_count'] > 0].merge(ext, on='addr', how='left')
    feat[['blend_in_count','blend_out_count','blend_net_value','LP_count','DeLP_count','unique_interactions','ratio']] = \
        feat[['blend_in_count','blend_out_count','blend_net_value','LP_count','DeLP_count','unique_interactions','ratio']].fillna(0)
    feat['label'] = feat['addr'].isin(targets).astype(int)
    return feat

feature_cols = [
    'buy_count','buy_value','buy_collections','sell_count','sell_value','sell_ratio',
    'total_trade_count','pnl_proxy','wallet_age_days','days_since_last_buy','recent_activity','tx_count',
    'blend_in_count','blend_out_count','blend_net_value','LP_count','DeLP_count','unique_interactions','ratio'
]

# Build T-30 dataset
print(f"Building T-30 dataset (cutoff: {ts_label(CUTOFF_T30)})...")
txs_sub = txs[txs['ts'] < CUTOFF_T30]
feat = compute_features(txs_sub, CUTOFF_T30)
feat_cols = [c for c in feature_cols if c in feat.columns]

X = feat[feat_cols].values.astype(np.float32)
y = feat['label'].values
print(f"  {len(feat):,} addresses, {y.sum():,} Sybil ({y.mean()*100:.1f}%)")

# 5-fold CV with full proba collection
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_proba = np.zeros(len(y))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    dval = lgb.Dataset(X[val_idx], label=y[val_idx], reference=dtrain)
    params = {
        'objective': 'binary', 'metric': 'auc',
        'learning_rate': 0.05, 'num_leaves': 63,
        'min_child_samples': 20,
        'scale_pos_weight': (y == 0).sum() / max((y == 1).sum(), 1),
        'verbose': -1
    }
    model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    all_proba[val_idx] = model.predict(X[val_idx])

# ── Threshold analysis ────────────────────────────────────────────────────────
auc = roc_auc_score(y, all_proba)
ap  = average_precision_score(y, all_proba)
precs, recs, thresholds = precision_recall_curve(y, all_proba)

# Find optimal thresholds
f1s = 2 * precs * recs / (precs + recs + 1e-10)
best_f1_idx = np.argmax(f1s[:-1])
best_f1_thresh = thresholds[best_f1_idx]

# Precision@90%: find threshold where precision >= 90%
p90_idx = np.where(precs[:-1] >= 0.90)[0]
p90_thresh = thresholds[p90_idx[0]] if len(p90_idx) > 0 else None

# Recall@80%: find threshold where recall >= 80%
r80_idx = np.where(recs[:-1] >= 0.80)[0]
r80_thresh = thresholds[r80_idx[-1]] if len(r80_idx) > 0 else None

print(f"\nT-30 Model Performance:")
print(f"  AUC = {auc:.4f}")
print(f"  Average Precision = {ap:.4f}")

print(f"\nThreshold Analysis:")
print(f"  Max F1 threshold = {best_f1_thresh:.4f}")
print(f"    Precision = {precs[best_f1_idx]:.4f}, Recall = {recs[best_f1_idx]:.4f}, F1 = {f1s[best_f1_idx]:.4f}")

if p90_thresh:
    p90_prec = precs[p90_idx[0]]
    p90_rec  = recs[p90_idx[0]]
    print(f"  Precision@90% threshold = {p90_thresh:.4f}")
    print(f"    Precision = {p90_prec:.4f}, Recall = {p90_rec:.4f}")
    flagged = (all_proba >= p90_thresh).sum()
    tp = ((all_proba >= p90_thresh) & (y == 1)).sum()
    print(f"    Would flag {flagged:,} addresses, catching {tp:,} true Sybils ({tp/y.sum()*100:.1f}%)")

if r80_thresh:
    r80_prec = precs[r80_idx[-1]]
    r80_rec  = recs[r80_idx[-1]]
    print(f"  Recall@80% threshold = {r80_thresh:.4f}")
    print(f"    Precision = {r80_prec:.4f}, Recall = {r80_rec:.4f}")
    flagged = (all_proba >= r80_thresh).sum()
    print(f"    Would flag {flagged:,} addresses")

# ── Summary table ────────────────────────────────────────────────────────────
thresh_results = []
for thresh in np.arange(0.1, 1.0, 0.05):
    preds = (all_proba >= thresh).astype(int)
    if preds.sum() == 0:
        break
    thresh_results.append({
        'threshold': round(thresh, 2),
        'flagged': preds.sum(),
        'precision': precision_score(y, preds, zero_division=0),
        'recall': recall_score(y, preds, zero_division=0),
        'f1': f1_score(y, preds, zero_division=0),
    })

thresh_df = pd.DataFrame(thresh_results)
thresh_df.to_csv(OUT_DIR / "threshold_analysis.csv", index=False)
print(f"\nThreshold sweep saved → {OUT_DIR}/threshold_analysis.csv")
print(thresh_df.to_string(index=False))

# ── Comparison vs baselines ───────────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON vs BASELINES")
print("="*60)
print(f"{'Model':<35} {'AUC':>6} {'Note'}")
print("-"*60)
print(f"{'ARTEMIS (post-hoc, GNN)':<35} {'0.803':>6}  Ref: WWW'24 paper")
print(f"{'Our T0 (all data)':<35} {'0.907':>6}  Pre-airdrop + extended feat")
print(f"{'Our T-7 (1 week before)':<35} {'0.906':>6}")
print(f"{'Our T-14 (2 weeks before)':<35} {'0.905':>6}")
print(f"{'Our T-30 (1 month before)':<35} {'0.904':>6}  ← MAIN RESULT")
print(f"{'Our T-60 (2 months before)':<35} {'0.903':>6}")
print(f"{'Our T-90 (3 months before)':<35} {'0.901':>6}")

# ── Plot ─────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PR Curve
    ax1 = axes[0]
    ax1.plot(recs[:-1], precs[:-1], color='#3B82F6', linewidth=2.5, label=f'T-30 model (AP={ap:.3f})')
    ax1.axhline(y.mean(), color='gray', linestyle='--', label=f'Baseline (Sybil rate {y.mean()*100:.1f}%)')

    if p90_thresh:
        ax1.plot(p90_rec, p90_prec, 'r*', markersize=15, label=f'Prec@90% (thresh={p90_thresh:.2f})')
    ax1.plot(recs[best_f1_idx], precs[best_f1_idx], 'g^', markersize=12,
             label=f'Max F1={f1s[best_f1_idx]:.3f} (thresh={best_f1_thresh:.2f})')

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve\nT-30 Pre-Airdrop Detection', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # AUC comparison bars
    ax2 = axes[1]
    models = ['ARTEMIS\n(post-hoc)', 'T-90\n(3mo before)', 'T-60\n(2mo)', 'T-30\n(1mo)', 'T-14\n(2wk)', 'T-7\n(1wk)', 'T0\n(airdrop day)']
    aucs   = [0.803,                  0.901,               0.903,          0.904,          0.905,          0.906,      0.907]
    colors = ['#94A3B8'] + ['#3B82F6'] * 4 + ['#10B981'] * 2

    bars = ax2.bar(range(len(models)), aucs, color=colors, edgecolor='white', linewidth=1.5)
    ax2.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='AUC=0.9 threshold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_ylim([0.75, 0.95])
    ax2.set_title('AUC vs Detection Window\n(Pre-airdrop vs ARTEMIS baseline)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    blue_patch = mpatches.Patch(color='#3B82F6', label='Our model (pre-airdrop)')
    gray_patch = mpatches.Patch(color='#94A3B8', label='ARTEMIS baseline')
    ax2.legend(handles=[gray_patch, blue_patch], fontsize=10, loc='lower right')

    plt.tight_layout()
    fig.savefig(OUT_DIR / "pr_and_comparison_plot.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {OUT_DIR}/pr_and_comparison_plot.png")
    plt.close()

except Exception as e:
    print(f"Plot error: {e}")
    import traceback; traceback.print_exc()

print("\nDone!")
