"""
04_feature_importance.py — Feature Importance Analysis

Analyze which features drive detection at each temporal window.
Compare importance ranking across T0, T-30, T-60, T-90.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset")
TXS_PATH = DATA_DIR / "TXS2_1662_1861.csv"
FEATURES_PATH = DATA_DIR / "addresses_all_with_loaylty_blend_blur_label319.csv"
OUT_DIR = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

T0 = 1700525735

WINDOWS = {
    "T0":   T0,
    "T-30": T0 - 30 * 86400,
    "T-60": T0 - 60 * 86400,
    "T-90": T0 - 90 * 86400,
}

FEATURE_LABELS = {
    'buy_count':          'Buy Count',
    'buy_value':          'Buy Volume (ETH)',
    'buy_collections':    'NFT Collection Diversity',
    'sell_count':         'Sell Count',
    'sell_value':         'Sell Volume (ETH)',
    'sell_ratio':         'Sell Ratio',
    'total_trade_count':  'Total Trade Count',
    'pnl_proxy':          'PnL Proxy (sell-buy)',
    'wallet_age_days':    'Wallet Age (days)',
    'days_since_last_buy':'Recency (days since last buy)',
    'recent_activity':    'Active in Last 30 Days',
    'tx_count':           'Total Tx Count',
    'blend_in_count':     'Blend Borrow Count',
    'blend_out_count':    'Blend Repay Count',
    'blend_net_value':    'Blend Net Value',
    'LP_count':           'LP Activity Count',
    'DeLP_count':         'DeLP Activity Count',
    'unique_interactions':'Unique Interactions',
    'ratio':              'NFT Volume Ratio',
}

def ts_label(ts_unix):
    return datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d")

print("=== Feature Importance Analysis ===\n")
print("Loading data...")

# Load targets
_flags = pd.read_csv(DATA_DIR / "airdrop_targets_behavior_flags.csv")
targets = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
del _flags
print(f"  Targets: {len(targets):,}")

# Load transactions
print("  Loading TXS2 (1.83GB)...")
txs = pd.read_csv(TXS_PATH, usecols=[
    'from', 'send', 'receive', 'timestamp', 'trade_price', 'event_type', 'contract_address'
], dtype={'timestamp': 'Int64', 'trade_price': 'float32'}, low_memory=False)
txs['ts'] = txs['timestamp'] // 1000
txs['from'] = txs['from'].str.lower().fillna('')
txs['send'] = txs['send'].str.lower().fillna('')
txs['receive'] = txs['receive'].str.lower().fillna('')
print(f"  {len(txs):,} transactions loaded")

# Load extended features
ext = pd.read_csv(FEATURES_PATH)
ext['address'] = ext['address'].str.lower()
ext_cols = ['address', 'blend_in_count', 'blend_out_count', 'blend_net_value',
            'LP_count', 'DeLP_count', 'unique_interactions', 'ratio']
ext = ext[ext_cols].rename(columns={'address': 'addr'})
print(f"  {len(ext):,} extended feature rows\n")

def compute_features(txs_sub, cutoff_ts):
    df = txs_sub.copy()
    buys = df[df['event_type'] == 'Sale'][['send', 'contract_address', 'trade_price', 'ts']].rename(columns={'send': 'addr'})
    sells = df[df['event_type'] == 'Sale'][['receive', 'contract_address', 'trade_price', 'ts']].rename(columns={'receive': 'addr'})

    all_addrs = pd.concat([buys[['addr']], sells[['addr']], df[['from']].rename(columns={'from': 'addr'})]).drop_duplicates()
    all_addrs = all_addrs[all_addrs['addr'].str.startswith('0x')]

    buy_stats = buys.groupby('addr').agg(
        buy_count=('addr', 'count'),
        buy_value=('trade_price', 'sum'),
        buy_collections=('contract_address', 'nunique'),
        buy_last_ts=('ts', 'max'),
        buy_first_ts=('ts', 'min'),
    ).reset_index()

    sell_stats = sells.groupby('addr').agg(
        sell_count=('addr', 'count'),
        sell_value=('trade_price', 'sum'),
    ).reset_index()

    from_stats = df.groupby('from').agg(
        tx_count=('ts', 'count'),
        first_tx_ts=('ts', 'min'),
    ).reset_index().rename(columns={'from': 'addr'})

    feat = all_addrs.merge(buy_stats, on='addr', how='left')
    feat = feat.merge(sell_stats, on='addr', how='left')
    feat = feat.merge(from_stats, on='addr', how='left')
    feat = feat.fillna(0)

    feat['total_trade_count'] = feat['buy_count'] + feat['sell_count']
    feat['sell_ratio'] = feat['sell_count'] / (feat['total_trade_count'] + 1e-6)
    feat['pnl_proxy'] = feat['sell_value'] - feat['buy_value']
    feat['wallet_age_days'] = (cutoff_ts - feat['first_tx_ts'].clip(lower=0)) / 86400
    feat['days_since_last_buy'] = (cutoff_ts - feat['buy_last_ts'].clip(lower=0)) / 86400
    feat['recent_activity'] = (feat['buy_last_ts'] > (cutoff_ts - 30*86400)).astype(int)

    feat = feat[feat['total_trade_count'] > 0].copy()
    feat = feat.merge(ext, on='addr', how='left')
    ext_fill = ['blend_in_count', 'blend_out_count', 'blend_net_value',
                'LP_count', 'DeLP_count', 'unique_interactions', 'ratio']
    feat[ext_fill] = feat[ext_fill].fillna(0)
    feat['label'] = feat['addr'].isin(targets).astype(int)
    return feat

# ── Analysis ─────────────────────────────────────────────────────────────────
feature_cols = list(FEATURE_LABELS.keys())
all_importance = {}

for window_name, cutoff_ts in WINDOWS.items():
    print(f"{'='*50}")
    print(f"Window: {window_name} ({ts_label(cutoff_ts)})")

    txs_sub = txs[txs['ts'] < cutoff_ts]
    feat = compute_features(txs_sub, cutoff_ts)
    feat_cols = [c for c in feature_cols if c in feat.columns]

    X = feat[feat_cols].values.astype(np.float32)
    y = feat['label'].values

    n_sybil = y.sum()
    print(f"  {len(feat):,} addresses | {n_sybil:,} Sybil ({n_sybil/len(feat)*100:.1f}%)")

    # Train single full model for importance (faster)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    importance_sum = np.zeros(len(feat_cols))
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx],
                             feature_name=feat_cols)
        dval = lgb.Dataset(X[val_idx], label=y[val_idx], reference=dtrain)

        params = {
            'objective': 'binary', 'metric': 'auc',
            'learning_rate': 0.05, 'num_leaves': 63,
            'min_child_samples': 20,
            'scale_pos_weight': (y == 0).sum() / max((y == 1).sum(), 1),
            'verbose': -1
        }
        model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        importance_sum += model.feature_importance(importance_type='gain')
        auc = roc_auc_score(y[val_idx], model.predict(X[val_idx]))
        fold_aucs.append(auc)

    mean_auc = np.mean(fold_aucs)
    print(f"  AUC={mean_auc:.4f}")

    # Normalize
    importance_norm = importance_sum / importance_sum.sum() * 100
    imp_df = pd.DataFrame({'feature': feat_cols, 'importance': importance_norm})
    imp_df = imp_df.sort_values('importance', ascending=False)
    imp_df['label'] = imp_df['feature'].map(FEATURE_LABELS)

    all_importance[window_name] = imp_df
    print("  Top 5 features:")
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['importance']:5.1f}%  {row['label']}")

# ── Compare importance across windows ────────────────────────────────────────
print("\n" + "="*70)
print("FEATURE IMPORTANCE COMPARISON")
print("="*70)

# Create comparison table
cmp = pd.DataFrame({'feature': list(FEATURE_LABELS.keys())})
cmp['label'] = cmp['feature'].map(FEATURE_LABELS)
for w in WINDOWS:
    if w in all_importance:
        imp = all_importance[w].set_index('feature')['importance']
        cmp[w] = cmp['feature'].map(imp).fillna(0)

cmp['avg'] = cmp[[w for w in WINDOWS if w in all_importance]].mean(axis=1)
cmp = cmp.sort_values('avg', ascending=False)

print(f"\n{'Feature':<30} {'T0':>6} {'T-30':>6} {'T-60':>6} {'T-90':>6}")
print("-" * 58)
for _, row in cmp.iterrows():
    print(f"{row['label']:<30} {row.get('T0', 0):>5.1f}% {row.get('T-30', 0):>5.1f}% {row.get('T-60', 0):>5.1f}% {row.get('T-90', 0):>5.1f}%")

cmp.to_csv(OUT_DIR / "feature_importance_comparison.csv", index=False)
print(f"\nSaved → {OUT_DIR}/feature_importance_comparison.csv")

# ── Plot ─────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    windows_list = list(WINDOWS.keys())
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    for ax, (window_name, color) in zip(axes.flatten(), zip(windows_list, colors)):
        if window_name not in all_importance:
            continue
        imp = all_importance[window_name].head(10).copy()
        imp['label'] = imp['feature'].map(FEATURE_LABELS)
        bars = ax.barh(range(len(imp)), imp['importance'], color=color, alpha=0.85)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp['label'], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (%)', fontsize=11)
        ax.set_title(f'{window_name} ({ts_label(WINDOWS[window_name])})', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, imp['importance'])):
            ax.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9)

    plt.suptitle('Top 10 Feature Importance by Pre-Airdrop Window\n(LightGBM, 5-fold CV, Blur Season 2)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_plot.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved → {OUT_DIR}/feature_importance_plot.png")
    plt.close()

    # ── Stability plot: how importance changes over time ──────────────────────
    top_features = cmp.head(6)['feature'].tolist()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    days_list = [0, 30, 60, 90]
    windows_order = ['T0', 'T-30', 'T-60', 'T-90']
    cmaps = plt.cm.tab10.colors

    for i, feat in enumerate(top_features):
        vals = []
        for w in windows_order:
            if w in all_importance:
                imp = all_importance[w].set_index('feature')['importance']
                vals.append(imp.get(feat, 0))
            else:
                vals.append(0)
        ax2.plot(days_list, vals, 'o-', color=cmaps[i], linewidth=2,
                 markersize=8, label=FEATURE_LABELS.get(feat, feat))

    ax2.set_xlabel('Days Before Airdrop', fontsize=12)
    ax2.set_ylabel('Feature Importance (%)', fontsize=12)
    ax2.set_title('Feature Importance Stability Over Time\n(Top 6 features across temporal windows)',
                  fontsize=13, fontweight='bold')
    ax2.invert_xaxis()
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "feature_stability_plot.png", dpi=150, bbox_inches='tight')
    print(f"Stability plot saved → {OUT_DIR}/feature_stability_plot.png")
    plt.close()

except Exception as e:
    print(f"Plot error: {e}")
    import traceback; traceback.print_exc()

print("\nDone!")
