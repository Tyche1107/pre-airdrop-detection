"""
17_common_features_lopo.py
Build protocol-agnostic feature set and redo LOPO cross-protocol experiment.
Common features: tx_count, total_volume, wallet_age_days, unique_contracts, active_span_days
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

OUT_DIR = '/Users/adelinewen/Desktop/pre-airdrop-detection/data'

BLUR_T0 = 1700525735
HOP_T0  = 1649030400
GIT_T0  = 1663804800

def load_blur_generic():
    txs = pd.read_csv(
        '/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset/TXS2_1662_1861.csv',
        usecols=['send', 'trade_value', 'timestamp', 'contract_address']
    )
    txs = txs[txs['timestamp'] < BLUR_T0 * 1000]
    txs['address'] = txs['send'].str.lower()
    def parse_eth(v):
        try:
            if isinstance(v, str) and v.startswith('0x'):
                return int(v, 16) / 1e18
            return float(v) / 1e18
        except:
            return 0.0
    txs['eth'] = txs['trade_value'].apply(parse_eth)
    g = txs.groupby('address')
    feat = pd.DataFrame({
        'address': list(g.groups.keys()),
        'tx_count': g['timestamp'].count().values,
        'total_volume': g['eth'].sum().values,
        'unique_contracts': g['contract_address'].nunique().values,
        'wallet_age_days': g['timestamp'].apply(lambda t: (BLUR_T0*1000 - t.min()) / 86400000).values,
        'active_span_days': g['timestamp'].apply(lambda t: (t.max() - t.min()) / 86400000).values,
    })
    _flags = pd.read_csv('/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset/airdrop_targets_behavior_flags.csv')
    sybil_set = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
    del _flags
    feat['is_sybil'] = feat['address'].isin(sybil_set).astype(int)
    feat['protocol'] = 'blur'
    print(f"Blur generic: {len(feat)} rows, sybil_rate={feat.is_sybil.mean():.3f}")
    return feat

def load_hop_generic():
    meta  = pd.read_csv('/Users/adelinewen/Desktop/dataset/hop/metadata.csv')
    ts    = pd.read_csv('/Users/adelinewen/Desktop/dataset/hop/timestamps.csv')
    sybil = pd.read_csv('/Users/adelinewen/Desktop/dataset/hop/sybil_addresses.csv')
    df = meta.rename(columns={'totalTxs': 'tx_count', 'totalVolume': 'total_volume'})
    df = df.merge(ts, on='address', how='left')
    sybil_set = set(sybil['address'].str.lower())
    df['is_sybil'] = df['address'].str.lower().isin(sybil_set).astype(int)
    df['wallet_age_days'] = (HOP_T0 - df['first_ts'].fillna(HOP_T0)) / 86400
    df['unique_contracts'] = 1
    df['active_span_days'] = 0
    df['protocol'] = 'hop'
    print(f"Hop generic: {len(df)} rows, sybil_rate={df.is_sybil.mean():.3f}")
    return df[['address','tx_count','total_volume','unique_contracts','wallet_age_days','active_span_days','is_sybil','protocol']]

def load_gitcoin_generic():
    df = pd.read_csv('/Users/adelinewen/Desktop/dataset/gitcoin/onchain_features.csv')
    df['total_volume'] = df['eth_sent'].astype(float) + df['eth_received'].astype(float)
    df['active_span_days'] = (df['last_tx_ts'].fillna(df['first_tx_ts']) - df['first_tx_ts'].fillna(0)) / 86400
    df['protocol'] = 'gitcoin'
    print(f"Gitcoin generic: {len(df)} rows, sybil_rate={df.is_sybil.mean():.3f}")
    return df[['address','tx_count','total_volume','unique_contracts','wallet_age_days','active_span_days','is_sybil','protocol']]

FEATURES = ['tx_count','total_volume','unique_contracts','wallet_age_days','active_span_days']

def preprocess(df):
    return df[FEATURES].fillna(0).values.astype(float), df['is_sybil'].values

def train_predict(X_tr, y_tr, X_te):
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                            class_weight='balanced', random_state=42, verbose=-1)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:,1]

print("Loading data...")
blur    = load_blur_generic()
hop     = load_hop_generic()
gitcoin = load_gitcoin_generic()

datasets = {'blur': blur, 'hop': hop, 'gitcoin': gitcoin}
results = []

pairs = [('hop+gitcoin','blur'),('blur+gitcoin','hop'),('blur+hop','gitcoin')]
for train_name, test_name in pairs:
    train_df = pd.concat([v for k,v in datasets.items() if k in train_name], ignore_index=True)
    test_df  = datasets[test_name]
    X_tr,y_tr = preprocess(train_df)
    X_te,y_te = preprocess(test_df)
    auc = roc_auc_score(y_te, train_predict(X_tr,y_tr,X_te))
    print(f"Zero-shot {train_name}->{test_name}: AUC {auc:.4f}")
    results.append({'train':train_name,'test':test_name,'auc':auc,'type':'zero-shot','feature_set':'common'})

for name, df in datasets.items():
    X,y = preprocess(df)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = [roc_auc_score(y[te], train_predict(X[tr],y[tr],X[te])) for tr,te in kf.split(X,y)]
    auc = np.mean(aucs)
    print(f"In-domain {name}: AUC {auc:.4f}")
    results.append({'train':f'{name}_indomain','test':name,'auc':auc,'type':'in-domain','feature_set':'common'})

res_df = pd.DataFrame(results)
res_df.to_csv(f'{OUT_DIR}/common_features_lopo.csv', index=False)
print(res_df.to_string(index=False))
print(f"Saved -> {OUT_DIR}/common_features_lopo.csv")
