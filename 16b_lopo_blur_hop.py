"""
Script 16b: Blur ↔ Hop LOPO (clean 2-protocol cross-validation)
Gitcoin excluded: application-layer Sybil behavior incompatible with on-chain protocols
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BLUR_FEAT  = '/Users/adelinewen/Desktop/dataset/blurtx/dataset/addresses_all_with_loaylty_blend_blur_label319.csv'
BLUR_SYBIL = '/Users/adelinewen/Desktop/dataset/blurtx/dataset/airdrop2_targets.txt'
HOP_META   = '/Users/adelinewen/Desktop/dataset/hop/metadata.csv'
HOP_SYBIL  = '/Users/adelinewen/Desktop/dataset/hop/sybil_addresses.csv'
HOP_TS     = '/Users/adelinewen/Desktop/dataset/hop/timestamps.csv'
BLUR_T0    = 1700525735
HOP_T0     = 1649030400
FEATS      = ['tx_count', 'volume', 'unique_interact', 'wallet_age_days']
OUT        = '/Users/adelinewen/Desktop/pre-airdrop-detection/data/lopo_blur_hop.csv'

def load_blur():
    df = pd.read_csv(BLUR_FEAT)
    sybil_set = set(open(BLUR_SYBIL).read().strip().lower().split('\n'))
    df['is_sybil'] = df['address'].str.lower().isin(sybil_set).astype(int)
    txs = pd.read_csv('/Users/adelinewen/Desktop/dataset/blurtx/dataset/TXS2_1662_1861.csv',
                      usecols=['to','send','timestamp'])
    txs['address'] = txs['to'].str.lower()
    first_ts = txs.groupby('address')['timestamp'].min().reset_index()
    first_ts.columns = ['address_lower','first_ts']
    df['address_lower'] = df['address'].str.lower()
    df = df.merge(first_ts, left_on='address_lower', right_on='address_lower', how='left')
    df['wallet_age_days'] = (BLUR_T0 - df['first_ts'].fillna(BLUR_T0)) / 86400
    df['wallet_age_days'] = df['wallet_age_days'].clip(lower=0)
    feat = pd.DataFrame({
        'address': df['address'], 'is_sybil': df['is_sybil'],
        'tx_count': df['buy_count'].fillna(0) + df['sell_count'].fillna(0),
        'volume': df['buy_value'].fillna(0) + df['sell_value'].fillna(0),
        'unique_interact': df['unique_interactions'].fillna(0),
        'wallet_age_days': df['wallet_age_days'].fillna(0),
        'protocol': 'blur'
    })
    print(f"Blur: {len(feat)} addrs, {feat['is_sybil'].sum()} sybils ({feat['is_sybil'].mean()*100:.1f}%)")
    return feat

def load_hop():
    meta  = pd.read_csv(HOP_META)
    sybs  = pd.read_csv(HOP_SYBIL, header=None, names=['address'])
    ts    = pd.read_csv(HOP_TS)
    sybil_set = set(sybs['address'].str.lower())
    meta['is_sybil'] = meta['address'].str.lower().isin(sybil_set).astype(int)
    meta = meta.merge(ts.rename(columns={'address':'address'}), on='address', how='left')
    meta['wallet_age_days'] = (HOP_T0 - meta['first_ts'].fillna(HOP_T0)) / 86400
    meta['wallet_age_days'] = meta['wallet_age_days'].clip(lower=0)
    feat = pd.DataFrame({
        'address': meta['address'], 'is_sybil': meta['is_sybil'],
        'tx_count': meta['totalTxs'].fillna(0),
        'volume': meta['totalVolume'].fillna(0),
        'unique_interact': 0,
        'wallet_age_days': meta['wallet_age_days'].fillna(0),
        'protocol': 'hop'
    })
    print(f"Hop: {len(feat)} addrs, {feat['is_sybil'].sum()} sybils ({feat['is_sybil'].mean()*100:.1f}%)")
    return feat

def preprocess(df):
    X = df[FEATS].copy().astype(float)
    X = X.apply(np.log1p)
    return X.values, df['is_sybil'].values

def precision_at_k(y_true, y_prob, k=1000):
    idx = np.argsort(y_prob)[::-1][:k]
    return y_true[idx].mean()

print("=" * 55)
print("LOPO: Blur ↔ Hop (Clean Cross-Protocol)")
print("=" * 55)

print("\nLoading datasets...")
blur = load_blur()
hop  = load_hop()

print("\n--- Zero-Shot Cross-Protocol ---")
results = []

# Blur → Hop
X_train, y_train = preprocess(blur)
X_test,  y_test  = preprocess(hop)
m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                        class_weight='balanced', random_state=42, verbose=-1)
m.fit(X_train, y_train)
prob = m.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, prob)
p_at_k = precision_at_k(y_test, prob, k=1000)
print(f"  Train: Blur → Test: Hop   | AUC: {auc:.4f} | P@1000: {p_at_k:.3f}")
results.append({'train':'blur','test':'hop','auc':auc,'p_at_1000':p_at_k,'type':'zero-shot'})

# Hop → Blur
X_train, y_train = preprocess(hop)
X_test,  y_test  = preprocess(blur)
m2 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                         class_weight='balanced', random_state=42, verbose=-1)
m2.fit(X_train, y_train)
prob2 = m2.predict_proba(X_test)[:,1]
auc2 = roc_auc_score(y_test, prob2)
p_at_k2 = precision_at_k(y_test, prob2, k=1000)
print(f"  Train: Hop  → Test: Blur  | AUC: {auc2:.4f} | P@1000: {p_at_k2:.3f}")
results.append({'train':'hop','test':'blur','auc':auc2,'p_at_1000':p_at_k2,'type':'zero-shot'})

# Fine-tuning: 1%, 5%, 10% of target
print("\n--- Fine-tuning Experiment (Train Blur → Fine-tune on Hop) ---")
np.random.seed(42)
for pct in [0.01, 0.05, 0.10, 0.20]:
    n = int(len(hop) * pct)
    idx = np.random.choice(len(hop), n, replace=False)
    mask = np.zeros(len(hop), dtype=bool); mask[idx] = True
    X_hop_train, y_hop_train = preprocess(hop.iloc[idx])
    X_hop_test,  y_hop_test  = preprocess(hop.iloc[~mask])
    X_combined = np.vstack([X_train, X_hop_train])  # X_train is blur
    y_combined = np.concatenate([y_train, y_hop_train])
    mf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             class_weight='balanced', random_state=42, verbose=-1)
    mf.fit(X_combined, y_combined)
    prob_ft = mf.predict_proba(X_hop_test)[:,1]
    auc_ft = roc_auc_score(y_hop_test, prob_ft)
    p_at_k_ft = precision_at_k(y_hop_test, prob_ft, k=min(1000, len(y_hop_test)))
    print(f"  Fine-tune {int(pct*100):3d}% Hop | AUC: {auc_ft:.4f} | P@1000: {p_at_k_ft:.3f}")
    results.append({'train':f'blur+hop{int(pct*100)}pct','test':'hop','auc':auc_ft,'p_at_1000':p_at_k_ft,'type':'fine-tune'})

# In-domain baselines
print("\n--- In-Domain Baselines (5-fold CV) ---")
for name, df in [('blur', blur), ('hop', hop)]:
    X, y = preprocess(df)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, prec = [], []
    for tr, te in kf.split(X, y):
        m_cv = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                   class_weight='balanced', random_state=42, verbose=-1)
        m_cv.fit(X[tr], y[tr])
        p = m_cv.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
        prec.append(precision_at_k(y[te], p, k=min(1000, len(y[te]))))
    print(f"  In-domain {name:5s}: AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f} | P@1000: {np.mean(prec):.3f}")
    results.append({'train':f'{name}_indomain','test':name,'auc':np.mean(aucs),'p_at_1000':np.mean(prec),'type':'in-domain'})

res_df = pd.DataFrame(results)
res_df.to_csv(OUT, index=False)
print(f"\nSaved → {OUT}")
print("\n" + "=" * 55)
print(res_df.to_string(index=False))
