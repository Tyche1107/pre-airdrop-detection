"""
24_layerzero_lopo.py — LayerZero LOPO (Leave-One-Protocol-Out)

LayerZero is a cross-chain messaging protocol (same category as Hop: bridge).
Experiment:
  A) Zero-shot: train on Blur+Hop+Gitcoin (common features), test on LZ
  B) Same-class: train on Hop only (bridge→bridge), test on LZ
  C) Fine-tune: Blur+LZ 1%/5%/10% labels
  D) LZ in-domain (upper bound)
  E) All 3 prior protocols → LZ

Common features: tx_count, total_volume, wallet_age_days, unique_contracts, active_span_days
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUT_DIR   = '/Users/adelinewen/Desktop/pre-airdrop-detection/data'
DATA_DIR  = '/Users/adelinewen/Desktop/dataset'
BLUR_T0   = 1700525735
HOP_T0    = 1649030400
GIT_T0    = 1663804800

COMMON_FEATS = ['tx_count', 'total_volume', 'wallet_age_days', 'unique_contracts', 'active_span_days']
LGB_PARAMS   = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                    class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
SKF          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Load datasets ──────────────────────────────────────────────────────────────
def load_blur():
    txs = pd.read_csv(f'{DATA_DIR}/blurtx/dataset/blurtx/dataset/TXS2_1662_1861.csv',
                      usecols=['send', 'trade_value', 'timestamp', 'contract_address'])
    txs = txs[txs['timestamp'] < BLUR_T0 * 1000]
    txs['address'] = txs['send'].str.lower()
    def eth(v):
        try:
            return (int(v, 16) if isinstance(v, str) and v.startswith('0x') else float(v)) / 1e18
        except: return 0.0
    txs['eth'] = txs['trade_value'].apply(eth)
    g = txs.groupby('address')
    df = pd.DataFrame({
        'address': list(g.groups.keys()),
        'tx_count': g['timestamp'].count().values,
        'total_volume': g['eth'].sum().values,
        'unique_contracts': g['contract_address'].nunique().values,
        'wallet_age_days': g['timestamp'].apply(lambda t: (BLUR_T0*1000 - t.min()) / 86400000).values,
        'active_span_days': g['timestamp'].apply(lambda t: (t.max() - t.min()) / 86400000).values,
    })
    _flags = pd.read_csv(f'{DATA_DIR}/blurtx/dataset/blurtx/dataset/airdrop_targets_behavior_flags.csv')
    sybils = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
    del _flags
    df['is_sybil'] = df['address'].isin(sybils).astype(int)
    df['protocol'] = 'blur'
    return df[COMMON_FEATS + ['address', 'is_sybil', 'protocol']].reset_index(drop=True)

def load_hop():
    meta  = pd.read_csv(f'{DATA_DIR}/hop/metadata.csv')
    ts    = pd.read_csv(f'{DATA_DIR}/hop/timestamps.csv')
    sybil = pd.read_csv(f'{DATA_DIR}/hop/sybil_addresses.csv')
    df = meta.rename(columns={'totalTxs': 'tx_count', 'totalVolume': 'total_volume'})
    df = df.merge(ts, on='address', how='left')
    df['wallet_age_days']  = (HOP_T0 - df['first_ts'].fillna(HOP_T0)) / 86400
    df['unique_contracts'] = 1
    df['active_span_days'] = 0.0
    df['is_sybil'] = df['address'].str.lower().isin(set(sybil['address'].str.lower())).astype(int)
    df['protocol'] = 'hop'
    return df[COMMON_FEATS + ['address', 'is_sybil', 'protocol']].reset_index(drop=True)

def load_gitcoin():
    feat = pd.read_csv(f'{DATA_DIR}/gitcoin/onchain_features.csv')
    # total_volume already exists; compute active_span_days from timestamps
    if 'first_tx_ts' in feat.columns and 'last_tx_ts' in feat.columns:
        feat['active_span_days'] = (feat['last_tx_ts'] - feat['first_tx_ts']).clip(0) / 86400
    for c in COMMON_FEATS:
        if c not in feat.columns:
            feat[c] = 0.0
    feat['protocol'] = 'gitcoin'
    return feat[COMMON_FEATS + ['address', 'is_sybil', 'protocol']].reset_index(drop=True)

def load_layerzero():
    df = pd.read_csv(f'{DATA_DIR}/layerzero/multichain_features.csv')
    df = df[df['total_tx'] > 0].copy()   # keep active addresses only
    df.rename(columns={'total_tx': 'tx_count'}, inplace=True)
    df['protocol'] = 'layerzero'
    for c in COMMON_FEATS:
        if c not in df.columns:
            df[c] = 0.0
    return df[COMMON_FEATS + ['address', 'is_sybil', 'protocol']].reset_index(drop=True)

print("Loading datasets...")
blur = load_blur()
hop  = load_hop()
lz   = load_layerzero()
print(f"Blur: {len(blur):,}  Hop: {len(hop):,}  LZ: {len(lz):,}")
print(f"LZ sybil rate: {lz['is_sybil'].mean():.3f}")

try:
    git = load_gitcoin()
    print(f"Gitcoin: {len(git):,}")
    HAS_GIT = True
except:
    print("Gitcoin: not available")
    HAS_GIT = False

def cv_auc(train_df, test_df):
    X_tr = train_df[COMMON_FEATS].fillna(0).values
    y_tr = train_df['is_sybil'].values
    X_te = test_df[COMMON_FEATS].fillna(0).values
    y_te = test_df['is_sybil'].values
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, preds)

def indomain_auc(df):
    X = df[COMMON_FEATS].fillna(0).values
    y = df['is_sybil'].values
    aucs = []
    for tr, val in SKF.split(X, y):
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
    return np.mean(aucs)

results = []

# A) Zero-shot: Blur+Hop+Gitcoin → LZ
print("\nA) Zero-shot: all prior protocols → LZ")
prior_list = [blur, hop, git] if HAS_GIT else [blur, hop]
train_all = pd.concat(prior_list, ignore_index=True)
auc_a = cv_auc(train_all, lz)
results.append({'experiment': 'zero-shot (blur+hop+gitcoin)', 'auc': round(auc_a, 4), 'type': 'zero-shot'})
print(f"   AUC = {auc_a:.4f}")

# B) Same-class bridge: Hop only → LZ
print("B) Same-class: Hop → LZ")
auc_b = cv_auc(hop, lz)
results.append({'experiment': 'same-class bridge (hop→lz)', 'auc': round(auc_b, 4), 'type': 'zero-shot'})
print(f"   AUC = {auc_b:.4f}")

# C) Blur only → LZ (baseline cross-protocol)
print("C) Blur only → LZ")
auc_c = cv_auc(blur, lz)
results.append({'experiment': 'cross-domain (blur→lz)', 'auc': round(auc_c, 4), 'type': 'zero-shot'})
print(f"   AUC = {auc_c:.4f}")

# D) Fine-tune: Blur + X% LZ labels
print("D) Fine-tune: Blur + LZ labels")
lz_sybil  = lz[lz['is_sybil'] == 1]
lz_normal = lz[lz['is_sybil'] == 0]
for pct in [0.01, 0.05, 0.10, 0.20]:
    n_s = max(1, int(len(lz_sybil)  * pct))
    n_n = max(1, int(len(lz_normal) * pct))
    lz_sample = pd.concat([
        lz_sybil.sample(n_s,  random_state=42),
        lz_normal.sample(n_n, random_state=42),
    ])
    train_ft = pd.concat([blur, lz_sample[COMMON_FEATS + ['address', 'is_sybil', 'protocol']]], ignore_index=True)
    auc_ft = cv_auc(train_ft, lz)
    label = f'fine-tune blur+lz_{int(pct*100)}pct'
    results.append({'experiment': label, 'auc': round(auc_ft, 4), 'type': 'fine-tune'})
    print(f"   Blur + LZ {int(pct*100)}%: AUC = {auc_ft:.4f}")

# E) LZ in-domain upper bound
print("E) LZ in-domain (5-fold CV)")
auc_e = indomain_auc(lz)
results.append({'experiment': 'lz in-domain', 'auc': round(auc_e, 4), 'type': 'in-domain'})
print(f"   AUC = {auc_e:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUT_DIR}/layerzero_lopo.csv', index=False)
print(f"\n{results_df.to_string(index=False)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
color_map = {'zero-shot': '#60a5fa', 'fine-tune': '#34d399', 'in-domain': '#f59e0b'}
bars = ax.barh(range(len(results_df)), results_df['auc'],
               color=[color_map[t] for t in results_df['type']], height=0.6)
ax.set_yticks(range(len(results_df)))
ax.set_yticklabels(results_df['experiment'], fontsize=10)
ax.set_xlabel('AUC')
ax.set_title('LayerZero LOPO: Protocol Transfer to Bridge Domain', fontweight='bold')
ax.axvline(0.803, color='#ef4444', linestyle='--', alpha=0.6, label='ARTEMIS 0.803')
ax.set_xlim(0, 1.05)
from matplotlib.patches import Patch
legend_elements = [Patch(color=c, label=t) for t, c in color_map.items()]
legend_elements.append(plt.Line2D([0], [0], color='#ef4444', linestyle='--', label='ARTEMIS 0.803'))
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
for bar, v in zip(ax.patches, results_df['auc']):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layerzero_lopo.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: layerzero_lopo.csv + layerzero_lopo.png")
