"""
00_build_nft_feats_labeled.py
Pre-compute labeled NFT feature files for T7, T30, T90 windows.
Saves: data/nft_feats_labeled_T7.csv, nft_feats_labeled_T30.csv, nft_feats_labeled_T90.csv
Each file has all behavioral features + 'is_sybil' label.

Ground truth: airdrop_targets_behavior_flags.csv (bw/ml/fd/hf flags)
Sybil = any flag == 1 (9,817 addresses)
Normal = all flags == 0 (43,665 airdrop recipients + all other Blur users)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset")
OUT_DIR  = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
OUT_DIR.mkdir(exist_ok=True)
T0 = 1700525735  # Nov 21, 2023 00:15:35 UTC

WINDOWS = {
    'T7':  T0 - 7  * 86400,
    'T30': T0 - 30 * 86400,
    'T90': T0 - 90 * 86400,
}

# ── Correct ground truth: behavior flags ──────────────────────────────────
print("Loading correct ground truth (behavior flags)...")
flags = pd.read_csv(DATA_DIR / "airdrop_targets_behavior_flags.csv")
flags['address'] = flags['address'].str.lower()
sybil_set = set(flags[
    (flags['bw_flag']==1) | (flags['ml_flag']==1) |
    (flags['fd_flag']==1) | (flags['hf_flag']==1)
]['address'])
print(f"  Total airdrop recipients: {len(flags):,}")
print(f"  Sybil (any flag=1):       {len(sybil_set):,}")
print(f"  Normal (all flags=0):     {len(flags) - len(sybil_set):,}")

print("Loading TXS2 (may take ~30s)...")
txs = pd.read_csv(
    DATA_DIR / "TXS2_1662_1861.csv",
    usecols=['from', 'send', 'receive', 'timestamp', 'trade_price', 'event_type', 'contract_address'],
    dtype={'timestamp': 'Int64', 'trade_price': 'float32'},
    low_memory=False
)
txs['ts'] = txs['timestamp'] // 1000
for c in ['from', 'send', 'receive']:
    txs[c] = txs[c].str.lower().fillna('')
print(f"  Loaded {len(txs):,} rows")

print("Loading extended features...")
ext = pd.read_csv(DATA_DIR / "addresses_all_with_loaylty_blend_blur_label319.csv")
ext['address'] = ext['address'].str.lower()
ext = ext[['address', 'blend_in_count', 'blend_out_count', 'blend_net_value',
           'LP_count', 'DeLP_count', 'unique_interactions', 'ratio']].rename(columns={'address': 'addr'})
print(f"  Extended features: {len(ext):,} addresses")

def build_features(txs_sub, cutoff):
    buys = txs_sub[txs_sub['event_type'] == 'Sale'][['send', 'contract_address', 'trade_price', 'ts']].rename(columns={'send': 'addr'})
    sells = txs_sub[txs_sub['event_type'] == 'Sale'][['receive', 'contract_address', 'trade_price', 'ts']].rename(columns={'receive': 'addr'})
    all_a = pd.concat([buys[['addr']], sells[['addr']], txs_sub[['from']].rename(columns={'from': 'addr'})]).drop_duplicates()
    all_a = all_a[all_a['addr'].str.startswith('0x')]

    buy_s = buys.groupby('addr').agg(
        buy_count=('addr', 'count'),
        buy_value=('trade_price', 'sum'),
        buy_collections=('contract_address', 'nunique'),
        buy_last_ts=('ts', 'max'),
        buy_first_ts=('ts', 'min'),
    ).reset_index()

    sell_s = sells.groupby('addr').agg(
        sell_count=('addr', 'count'),
        sell_value=('trade_price', 'sum'),
    ).reset_index()

    from_s = txs_sub.groupby('from').agg(
        tx_count=('ts', 'count'),
        first_tx_ts=('ts', 'min'),
    ).reset_index().rename(columns={'from': 'addr'})

    feat = all_a.merge(buy_s, on='addr', how='left').merge(sell_s, on='addr', how='left').merge(from_s, on='addr', how='left').fillna(0)
    feat['total_trade_count'] = feat['buy_count'] + feat['sell_count']
    feat['sell_ratio'] = feat['sell_count'] / (feat['total_trade_count'] + 1e-6)
    feat['pnl_proxy'] = feat['sell_value'] - feat['buy_value']
    feat['wallet_age_days'] = (cutoff - feat['first_tx_ts'].clip(lower=0)) / 86400
    feat['days_since_last_buy'] = (cutoff - feat['buy_last_ts'].clip(lower=0)) / 86400
    feat['recent_activity'] = (feat['buy_last_ts'] > (cutoff - 30 * 86400)).astype(int)

    feat = feat[feat['total_trade_count'] > 0].copy()
    feat = feat.merge(ext, on='addr', how='left')
    for c in ['blend_in_count', 'blend_out_count', 'blend_net_value', 'LP_count', 'DeLP_count', 'unique_interactions', 'ratio']:
        if c in feat.columns:
            feat[c] = feat[c].fillna(0)

    # Correct label: any behavior flag == 1
    feat['is_sybil'] = feat['addr'].isin(sybil_set).astype(int)
    feat = feat.rename(columns={'addr': 'address'})
    return feat

for wname, cutoff in WINDOWS.items():
    dt = datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime('%Y-%m-%d')
    print(f"\nBuilding {wname} (cutoff {dt})...")
    sub = txs[txs['ts'] < cutoff]
    print(f"  Transactions: {len(sub):,}")
    feat = build_features(sub, cutoff)
    n_sybil = feat['is_sybil'].sum()
    print(f"  Addresses: {len(feat):,} | Sybil: {n_sybil:,} ({n_sybil/len(feat)*100:.1f}%)")
    out_path = OUT_DIR / f"nft_feats_labeled_{wname}.csv"
    feat.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

print("\nAll feature files built with correct ground truth.")
