"""
Script 16: Leave-One-Protocol-Out (LOPO) Cross-Protocol Validation
Train on 2 protocols → zero-shot predict the 3rd
Protocols: Blur (NFT), Hop (Bridge), Gitcoin (Public Goods)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
BLUR_FEAT   = '/Users/adelinewen/Desktop/dataset/blurtx/dataset/addresses_all_with_loaylty_blend_blur_label319.csv'
BLUR_SYBIL  = '/Users/adelinewen/Desktop/dataset/blurtx/dataset/airdrop2_targets.txt'
HOP_META    = '/Users/adelinewen/Desktop/dataset/hop/metadata.csv'
HOP_SYBIL   = '/Users/adelinewen/Desktop/dataset/hop/sybil_addresses.csv'
HOP_TS      = '/Users/adelinewen/Desktop/dataset/hop/timestamps.csv'
GIT_FEAT    = '/Users/adelinewen/Desktop/dataset/gitcoin/onchain_features.csv'
OUT_DIR     = '/Users/adelinewen/Desktop/pre-airdrop-detection/data'

# Airdrop dates for wallet age calculation
BLUR_T0     = 1700525735   # 2023-11-21
HOP_T0      = 1649030400   # 2022-04-04
GIT_T0      = 1663804800   # 2022-09-22 (GR15)

# ─── Load & engineer per protocol ────────────────────────────────────────────
def load_blur():
    df = pd.read_csv(BLUR_FEAT)
    sybil_set = set(open(BLUR_SYBIL).read().strip().lower().split('\n'))
    df['is_sybil'] = df['address'].str.lower().isin(sybil_set).astype(int)
    # Use TXS2 for wallet age via first transaction
    txs = pd.read_csv('/Users/adelinewen/Desktop/dataset/blurtx/dataset/TXS2_1662_1861.csv',
                      usecols=['to', 'send', 'timestamp'])
    txs['address'] = txs['to'].str.lower()
    first_ts = txs.groupby('address')['timestamp'].min().reset_index()
    first_ts.columns = ['address_lower', 'first_ts']
    df['address_lower'] = df['address'].str.lower()
    df = df.merge(first_ts, left_on='address_lower', right_on='address_lower', how='left')
    df['wallet_age_days'] = (BLUR_T0 - df['first_ts'].fillna(BLUR_T0)) / 86400
    df['wallet_age_days'] = df['wallet_age_days'].clip(lower=0)

    # Fix blend_net_value: stored in WEI, divide by 1e18 for ETH
    df['blend_net_value_eth'] = df['blend_net_value'].fillna(0) / 1e18

    feat = pd.DataFrame({
        'address':           df['address'],
        'is_sybil':          df['is_sybil'],
        'tx_count':          df['buy_count'].fillna(0) + df['sell_count'].fillna(0),
        'volume':            df['buy_value'].fillna(0) + df['sell_value'].fillna(0),
        'unique_interact':   df['unique_interactions'].fillna(0),
        'wallet_age_days':   df['wallet_age_days'].fillna(0),
        'protocol':          'blur'
        # Note: LP_count/LP_value/DeLP_count/DeLP_value excluded (all zeros)
    })
    print(f"Blur: {len(feat)} addrs, {feat['is_sybil'].sum()} sybils ({feat['is_sybil'].mean()*100:.1f}%)")
    return feat

def load_hop():
    meta   = pd.read_csv(HOP_META)
    sybils = pd.read_csv(HOP_SYBIL, header=None, names=['address'])
    ts     = pd.read_csv(HOP_TS)

    sybil_set = set(sybils['address'].str.lower())
    meta['is_sybil'] = meta['address'].str.lower().isin(sybil_set).astype(int)
    meta = meta.merge(ts.rename(columns={'address': 'address'}), on='address', how='left')
    meta['wallet_age_days'] = (HOP_T0 - meta['first_ts'].fillna(HOP_T0)) / 86400
    meta['wallet_age_days'] = meta['wallet_age_days'].clip(lower=0)

    feat = pd.DataFrame({
        'address':          meta['address'],
        'is_sybil':         meta['is_sybil'],
        'tx_count':         meta['totalTxs'].fillna(0),
        'volume':           meta['totalVolume'].fillna(0),
        'unique_interact':  0,   # not available for Hop
        'wallet_age_days':  meta['wallet_age_days'].fillna(0),
        'protocol':         'hop'
    })
    print(f"Hop: {len(feat)} addrs, {feat['is_sybil'].sum()} sybils ({feat['is_sybil'].mean()*100:.1f}%)")
    return feat

def load_gitcoin():
    df = pd.read_csv(GIT_FEAT)
    df = df.dropna(subset=['address'])

    # Merge ERC-20 token features if available (better signal for Gitcoin)
    erc20_path = '/Users/adelinewen/Desktop/dataset/gitcoin/erc20_features.csv'
    if os.path.exists(erc20_path):
        erc = pd.read_csv(erc20_path)
        erc = erc.dropna(subset=['address'])
        df = df.merge(erc[['address','token_tx_count','stable_send_count',
                            'stable_volume_usd','unique_token_contracts',
                            'gitcoin_donations']].rename(
                            columns={'address':'address'}),
                      on='address', how='left')
        df['token_tx_count'] = df['token_tx_count'].fillna(0)
        df['stable_volume_usd'] = df['stable_volume_usd'].fillna(0)
        df['unique_token_contracts'] = df['unique_token_contracts'].fillna(0)
        df['gitcoin_donations'] = df['gitcoin_donations'].fillna(0)
        # Use token tx as primary tx_count (more signal for Gitcoin)
        df['best_tx_count'] = df['token_tx_count'].clip(lower=df['tx_count'].fillna(0))
        df['best_volume']   = df['stable_volume_usd'].clip(lower=df['total_volume'].fillna(0))
        df['best_unique']   = df['unique_token_contracts'].clip(lower=df['unique_contracts'].fillna(0))
        print(f"  ERC-20 merged: {(df['token_tx_count']>0).sum()} addresses with token data")
    else:
        df['best_tx_count'] = df['tx_count'].fillna(0)
        df['best_volume']   = df['total_volume'].fillna(0)
        df['best_unique']   = df['unique_contracts'].fillna(0)

    feat = pd.DataFrame({
        'address':          df['address'],
        'is_sybil':         df['is_sybil'].fillna(0).astype(int),
        'tx_count':         df['best_tx_count'],
        'volume':           df['best_volume'],
        'unique_interact':  df['best_unique'],
        'wallet_age_days':  df['wallet_age_days'].fillna(0),
        'protocol':         'gitcoin'
    })
    print(f"Gitcoin: {len(feat)} addrs, {feat['is_sybil'].sum()} sybils ({feat['is_sybil'].mean()*100:.1f}%)")
    print(f"  Non-zero tx_count: {(feat['tx_count']>0).sum()} ({(feat['tx_count']>0).mean()*100:.1f}%)")
    return feat

# ─── Feature columns ─────────────────────────────────────────────────────────
FEATS = ['tx_count', 'volume', 'unique_interact', 'wallet_age_days']

def normalize_protocol(df):
    """Quantile-normalize each protocol's features independently (rank-based, scale-invariant)."""
    X = df[FEATS].copy().astype(float)
    qt = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_norm = pd.DataFrame(qt.fit_transform(X), columns=FEATS)
    return X_norm

def preprocess(df):
    X = df[FEATS].copy().astype(float)
    X = X.apply(np.log1p)
    return X.values, df['is_sybil'].values

def train_and_eval(train_df, test_df, train_name, test_name):
    X_train, y_train = preprocess(train_df)
    X_test,  y_test  = preprocess(test_df)

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        class_weight='balanced', random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Train: {train_name:20s} | Test: {test_name:10s} | AUC: {auc:.4f}")
    return auc

# ─── Main ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Leave-One-Protocol-Out (LOPO) Experiment")
print("=" * 60)

print("\nLoading datasets...")
blur    = load_blur()
hop     = load_hop()
gitcoin = load_gitcoin()

# Note: per-protocol quantile normalization NOT applied — distorts signal
# when Sybil rates differ significantly (e.g., Gitcoin 71% vs Blur 20%)
datasets = {'blur': blur, 'hop': hop, 'gitcoin': gitcoin}

print("\n--- LOPO Results ---")
results = []

for held_out in ['blur', 'hop', 'gitcoin']:
    train_protos = [p for p in ['blur', 'hop', 'gitcoin'] if p != held_out]
    train_df = pd.concat([datasets[p] for p in train_protos], ignore_index=True)
    test_df  = datasets[held_out]
    train_name = '+'.join(train_protos)
    auc = train_and_eval(train_df, test_df, train_name, held_out)
    results.append({'train': train_name, 'test': held_out, 'auc': auc, 'type': 'zero-shot'})

# Also: same-protocol in-domain baselines
print("\n--- In-Domain Baselines (5-fold CV) ---")
from sklearn.model_selection import StratifiedKFold

for name, df in datasets.items():
    X, y = preprocess(df)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, te_idx in kf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                               num_leaves=31, class_weight='balanced',
                               random_state=42, verbose=-1)
        m.fit(X[tr_idx], y[tr_idx])
        aucs.append(roc_auc_score(y[te_idx], m.predict_proba(X[te_idx])[:,1]))
    auc_mean = np.mean(aucs)
    print(f"  In-domain {name:10s}: AUC {auc_mean:.4f} (±{np.std(aucs):.4f})")
    results.append({'train': name+'_indomain', 'test': name, 'auc': auc_mean, 'type': 'in-domain'})

# Save results
res_df = pd.DataFrame(results)
out_path = f'{OUT_DIR}/lopo_results.csv'
res_df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
zero_shot = res_df[res_df['type'] == 'zero-shot']
print(f"Zero-shot AUC (avg): {zero_shot['auc'].mean():.4f}")
print(f"Zero-shot AUC (min): {zero_shot['auc'].min():.4f}")
print(f"Zero-shot AUC (max): {zero_shot['auc'].max():.4f}")
print()
print(res_df.to_string(index=False))
