"""
03_temporal_ablation.py — Temporal Ablation Study

T0 = 2023-11-21 00:15:35 UTC (first Blur Season 2 airdrop claim tx)

For each cutoff T_k (T-7, T-14, T-30, T-60, T-90 days before T0):
  1. Filter TXS2 transactions to timestamp < T_k
  2. Compute per-address behavioral features
  3. Label: in airdrop2_targets.txt → Sybil hunter (1) else normal (0)
  4. Train LightGBM with 5-fold CV
  5. Compute AUC, F1, Precision, Recall at optimal threshold

Result: AUC curve over pre-airdrop time windows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset")
TXS_PATH = DATA_DIR / "TXS2_1662_1861.csv"
TARGETS_PATH = DATA_DIR / "airdrop2_targets.txt"
FEATURES_PATH = DATA_DIR / "addresses_all_with_loaylty_blend_blur_label319.csv"
OUT_DIR = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── T0 and ablation windows ──────────────────────────────────────────────────
T0 = 1700525735  # Nov 21, 2023 00:15:35 UTC

WINDOWS = {
    "T0":    T0,
    "T-7":   T0 - 7  * 86400,   # Nov 14
    "T-14":  T0 - 14 * 86400,   # Nov  7
    "T-30":  T0 - 30 * 86400,   # Oct 22
    "T-60":  T0 - 60 * 86400,   # Sep 22
    "T-90":  T0 - 90 * 86400,   # Aug 22
}

def ts_label(ts_unix):
    return datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d")

print("=== Temporal Ablation: Pre-Airdrop Sybil Detection ===\n")
for k, v in WINDOWS.items():
    print(f"  {k:5s} → {ts_label(v)}")
print()

# ── Load airdrop targets (ground truth) ─────────────────────────────────────
print("Loading airdrop targets...")
with open(TARGETS_PATH) as f:
    targets = set(line.strip().lower() for line in f if line.strip())
print(f"  Airdrop hunters (any_flag ground truth): {len(targets):,}")

# ── Load & preprocess TXS2 ──────────────────────────────────────────────────
print("\nLoading TXS2 transactions (3.2M rows, this takes ~30s)...")
dtypes = {
    'from': str, 'to': str, 'send': str, 'receive': str,
    'timestamp': 'Int64', 'trade_price': float, 'gas_used': str,
    'contract_address': str, 'event_type': str, 'exchange_name': str
}
txs = pd.read_csv(TXS_PATH, usecols=[
    'from', 'to', 'send', 'receive',
    'timestamp', 'trade_price', 'event_type', 'contract_address', 'exchange_name'
], dtype={'timestamp': 'Int64', 'trade_price': 'float32'}, low_memory=False)

print(f"  Loaded {len(txs):,} rows")

# Convert timestamp: stored as milliseconds
txs['ts'] = txs['timestamp'] // 1000
txs['from'] = txs['from'].str.lower().fillna('')
txs['send'] = txs['send'].str.lower().fillna('')
txs['to'] = txs['to'].str.lower().fillna('')
txs['receive'] = txs['receive'].str.lower().fillna('')

# The transaction initiator is 'from' (for Buy = send pays, Sell = send receives)
# We use 'send' as the trader address (the one who initiated the trade)
# For Sales on Blur: send = seller, receive = buyer address
txs_time_range = (
    datetime.fromtimestamp(txs['ts'].min(), tz=timezone.utc).strftime("%Y-%m-%d"),
    datetime.fromtimestamp(txs['ts'].max(), tz=timezone.utc).strftime("%Y-%m-%d")
)
print(f"  TXS time range: {txs_time_range[0]} → {txs_time_range[1]}")

# ── Feature engineering function ─────────────────────────────────────────────
def compute_features(txs_sub: pd.DataFrame, cutoff_ts: int) -> pd.DataFrame:
    """
    Given a filtered transaction DataFrame (all ts < cutoff_ts),
    compute per-address behavioral features.
    """
    df = txs_sub.copy()

    # Buyer = 'send' address on a sale event; Seller = 'receive' address? 
    # On Blur: event_type='Sale', send=buyer, receive=seller
    # Let's pivot: identify trader as 'send' (buyer) or 'receive' (seller)
    buys = df[df['event_type'] == 'Sale'][['send', 'contract_address', 'trade_price', 'ts']].rename(columns={'send': 'addr'})
    buys['is_buy'] = 1

    sells = df[df['event_type'] == 'Sale'][['receive', 'contract_address', 'trade_price', 'ts']].rename(columns={'receive': 'addr'})
    sells['is_sell'] = 1

    # All addresses
    all_addrs = pd.concat([
        buys[['addr']],
        sells[['addr']],
        df[['from']].rename(columns={'from': 'addr'}),
        df[['to']].rename(columns={'to': 'addr'})
    ]).drop_duplicates()
    all_addrs = all_addrs[all_addrs['addr'].str.startswith('0x')]

    # --- Buy stats ---
    buy_stats = buys.groupby('addr').agg(
        buy_count=('is_buy', 'count'),
        buy_value=('trade_price', 'sum'),
        buy_collections=('contract_address', 'nunique'),
        buy_last_ts=('ts', 'max'),
        buy_first_ts=('ts', 'min'),
    ).reset_index()

    # --- Sell stats ---
    sell_stats = sells.groupby('addr').agg(
        sell_count=('is_sell', 'count'),
        sell_value=('trade_price', 'sum'),
        sell_last_ts=('ts', 'max'),
    ).reset_index()

    # --- All tx stats ---
    from_stats = df.groupby('from').agg(
        tx_count=('ts', 'count'),
        first_tx_ts=('ts', 'min'),
        last_tx_ts=('ts', 'max'),
    ).reset_index().rename(columns={'from': 'addr'})

    # --- Merge ---
    feat = all_addrs.merge(buy_stats, on='addr', how='left')
    feat = feat.merge(sell_stats, on='addr', how='left')
    feat = feat.merge(from_stats, on='addr', how='left')

    feat = feat.fillna(0)

    # --- Derived features ---
    feat['total_trade_count'] = feat['buy_count'] + feat['sell_count']
    feat['sell_ratio'] = feat['sell_count'] / (feat['total_trade_count'] + 1e-6)
    feat['pnl_proxy'] = feat['sell_value'] - feat['buy_value']
    feat['wallet_age_days'] = (cutoff_ts - feat['first_tx_ts'].clip(lower=0)) / 86400
    feat['days_since_last_buy'] = (cutoff_ts - feat['buy_last_ts'].clip(lower=0)) / 86400
    # Recency: how active in last 30 days relative to cutoff
    feat['recent_activity'] = (feat['buy_last_ts'] > (cutoff_ts - 30*86400)).astype(int)

    # Filter out non-trader addresses (tx_count == 0 and buy_count == 0)
    feat = feat[feat['total_trade_count'] > 0].copy()

    return feat

# ── Load extended feature file (precomputed) ─────────────────────────────────
print("\nLoading extended features (blend protocol etc.)...")
ext = pd.read_csv(FEATURES_PATH)
ext['address'] = ext['address'].str.lower()
ext_cols = ['address', 'blend_in_count', 'blend_out_count', 'blend_net_value',
            'LP_count', 'DeLP_count', 'unique_interactions', 'ratio']
ext = ext[ext_cols].rename(columns={'address': 'addr'})
print(f"  {len(ext):,} addresses with extended features")

# ── Main ablation loop ────────────────────────────────────────────────────────
results = []
feature_importances = {}

for window_name, cutoff_ts in WINDOWS.items():
    cutoff_dt = ts_label(cutoff_ts)
    print(f"\n{'='*60}")
    print(f"Window: {window_name} ({cutoff_dt})")

    # Filter transactions
    txs_sub = txs[txs['ts'] < cutoff_ts]
    n_txs = len(txs_sub)
    print(f"  Transactions before cutoff: {n_txs:,}")

    if n_txs < 1000:
        print("  SKIP — too few transactions")
        continue

    # Compute features
    feat = compute_features(txs_sub, cutoff_ts)

    # Merge extended features
    feat = feat.merge(ext, on='addr', how='left')
    ext_cols_fill = ['blend_in_count', 'blend_out_count', 'blend_net_value',
                     'LP_count', 'DeLP_count', 'unique_interactions', 'ratio']
    feat[ext_cols_fill] = feat[ext_cols_fill].fillna(0)

    # Label
    feat['label'] = feat['addr'].isin(targets).astype(int)
    n_total = len(feat)
    n_sybil = feat['label'].sum()
    sybil_rate = n_sybil / n_total * 100
    print(f"  Addresses: {n_total:,} | Sybil: {n_sybil:,} ({sybil_rate:.1f}%)")

    if n_sybil < 100:
        print("  SKIP — too few Sybil labels")
        continue

    # Features for model
    feature_cols = [
        'buy_count', 'buy_value', 'buy_collections',
        'sell_count', 'sell_value', 'sell_ratio',
        'total_trade_count', 'pnl_proxy',
        'wallet_age_days', 'days_since_last_buy', 'recent_activity',
        'tx_count',
        # Extended
        'blend_in_count', 'blend_out_count', 'blend_net_value',
        'LP_count', 'DeLP_count', 'unique_interactions', 'ratio'
    ]
    # Keep only cols that exist
    feature_cols = [c for c in feature_cols if c in feat.columns]

    X = feat[feature_cols].values.astype(np.float32)
    y = feat['label'].values

    # 5-fold CV
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    fold_f1s = []
    fold_precs = []
    fold_recs = []
    all_proba = np.zeros(len(y))
    all_y = y.copy()

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': (y == 0).sum() / max((y == 1).sum(), 1),
            'verbose': -1,
        }

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        )

        proba = model.predict(X_val)
        all_proba[val_idx] = proba

        auc = roc_auc_score(y_val, proba)
        # Optimal threshold via F1
        precs_arr, recs_arr, thresholds = precision_recall_curve(y_val, proba)
        f1s = 2 * precs_arr * recs_arr / (precs_arr + recs_arr + 1e-10)
        best_thresh = thresholds[np.argmax(f1s[:-1])] if len(thresholds) > 0 else 0.5
        preds = (proba >= best_thresh).astype(int)

        fold_aucs.append(auc)
        fold_f1s.append(f1_score(y_val, preds, zero_division=0))
        fold_precs.append(precision_score(y_val, preds, zero_division=0))
        fold_recs.append(recall_score(y_val, preds, zero_division=0))

    mean_auc  = np.mean(fold_aucs)
    mean_f1   = np.mean(fold_f1s)
    mean_prec = np.mean(fold_precs)
    mean_rec  = np.mean(fold_recs)

    print(f"  AUC={mean_auc:.4f} | F1={mean_f1:.4f} | Prec={mean_prec:.4f} | Rec={mean_rec:.4f}")

    results.append({
        'window': window_name,
        'cutoff': cutoff_dt,
        'days_before_T0': (T0 - cutoff_ts) // 86400,
        'n_addresses': n_total,
        'n_sybil': n_sybil,
        'sybil_rate_pct': sybil_rate,
        'auc': mean_auc,
        'f1': mean_f1,
        'precision': mean_prec,
        'recall': mean_rec,
    })

# ── Results summary ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEMPORAL ABLATION RESULTS")
print("="*70)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv(OUT_DIR / "temporal_ablation_results.csv", index=False)
print(f"\nSaved → {OUT_DIR}/temporal_ablation_results.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    days = results_df['days_before_T0'].values
    aucs = results_df['auc'].values
    f1s  = results_df['f1'].values

    ax1 = axes[0]
    ax1.plot(days, aucs, 'o-', color='#3B82F6', linewidth=2, markersize=8, label='AUC')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlabel('Days Before Airdrop (T0)', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('Pre-Airdrop Detection AUC\nvs. Observation Window', fontsize=13, fontweight='bold')
    ax1.invert_xaxis()
    ax1.set_ylim(0.4, 1.0)
    ax1.legend()
    ax1.grid(alpha=0.3)
    # Label each point
    for d, a, w in zip(days, aucs, results_df['window']):
        ax1.annotate(f'{w}\n{a:.3f}', (d, a), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    ax2 = axes[1]
    ax2.plot(days, f1s, 's-', color='#10B981', linewidth=2, markersize=8, label='F1')
    ax2.set_xlabel('Days Before Airdrop (T0)', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Pre-Airdrop Detection F1\nvs. Observation Window', fontsize=13, fontweight='bold')
    ax2.invert_xaxis()
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()
    ax2.grid(alpha=0.3)
    for d, f, w in zip(days, f1s, results_df['window']):
        ax2.annotate(f'{w}\n{f:.3f}', (d, f), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "temporal_ablation_plot.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved → {OUT_DIR}/temporal_ablation_plot.png")
    plt.close()
except Exception as e:
    print(f"Plot failed: {e}")

print("\nDone!")
