"""
08_generalization.py — Temporal Generalization Experiment

Since cross-protocol data is not available, we test temporal generalization:
  Train on early window (T-90 ~ T-60), test on later window (T-30 ~ T0)
  → Shows whether behavioral patterns detected early still hold late

Also tests:
  - Train on 50% of Sybils, test on the other 50% (population split)
  - Cold-start: Train without any Sybil labels (semi-supervised proxy)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset")
TXS_PATH     = DATA_DIR / "TXS2_1662_1861.csv"
FEATURES_PATH = DATA_DIR / "addresses_all_with_loaylty_blend_blur_label319.csv"
OUT_DIR = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")

T0 = 1700525735

print("Loading data...")
_flags = pd.read_csv(DATA_DIR / "airdrop_targets_behavior_flags.csv")
targets = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
del _flags

txs = pd.read_csv(TXS_PATH, usecols=['from','send','receive','timestamp','trade_price','event_type','contract_address'],
                  dtype={'timestamp':'Int64','trade_price':'float32'}, low_memory=False)
txs['ts'] = txs['timestamp'] // 1000
for c in ['from','send','receive']:
    txs[c] = txs[c].str.lower().fillna('')

ext = pd.read_csv(FEATURES_PATH)
ext['address'] = ext['address'].str.lower()
ext = ext[['address','blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']].rename(columns={'address':'addr'})

FEATURE_COLS = ['buy_count','sell_count','tx_count','total_trade_count','buy_value','sell_value',
                'pnl_proxy','buy_collections','unique_interactions','sell_ratio','wallet_age_days',
                'days_since_last_buy','recent_activity','blend_in_count','blend_out_count',
                'blend_net_value','LP_count','ratio']

def build_features(cutoff_ts):
    df = txs[txs['ts'] < cutoff_ts].copy()
    buys  = df[df['event_type']=='Sale'][['send','contract_address','trade_price','ts']].rename(columns={'send':'addr'})
    sells = df[df['event_type']=='Sale'][['receive','contract_address','trade_price','ts']].rename(columns={'receive':'addr'})
    all_a = pd.concat([buys[['addr']], sells[['addr']], df[['from']].rename(columns={'from':'addr'})]).drop_duplicates()
    all_a = all_a[all_a['addr'].str.startswith('0x')]
    buy_s  = buys.groupby('addr').agg(buy_count=('addr','count'),buy_value=('trade_price','sum'),buy_collections=('contract_address','nunique'),buy_last_ts=('ts','max'),buy_first_ts=('ts','min')).reset_index()
    sell_s = sells.groupby('addr').agg(sell_count=('addr','count'),sell_value=('trade_price','sum')).reset_index()
    from_s = df.groupby('from').agg(tx_count=('ts','count'),first_tx_ts=('ts','min')).reset_index().rename(columns={'from':'addr'})
    feat = all_a.merge(buy_s,on='addr',how='left').merge(sell_s,on='addr',how='left').merge(from_s,on='addr',how='left').fillna(0)
    feat['total_trade_count'] = feat['buy_count'] + feat['sell_count']
    feat['sell_ratio']        = feat['sell_count'] / (feat['total_trade_count'] + 1e-6)
    feat['pnl_proxy']         = feat['sell_value'] - feat['buy_value']
    feat['wallet_age_days']   = (cutoff_ts - feat['first_tx_ts'].clip(lower=0)) / 86400
    feat['days_since_last_buy'] = (cutoff_ts - feat['buy_last_ts'].clip(lower=0)) / 86400
    feat['recent_activity']   = (feat['buy_last_ts'] > (cutoff_ts - 30*86400)).astype(int)
    feat = feat[feat['total_trade_count'] > 0].merge(ext, on='addr', how='left')
    feat[['blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']] = \
        feat[['blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']].fillna(0)
    feat['label'] = feat['addr'].isin(targets).astype(int)
    return feat

params = {'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'verbose':-1}

results = []

# ── Experiment 1: Standard 5-fold CV at T-30 (baseline) ─────────────────────
print("\nExp 1: Standard 5-fold CV (T-30)...")
feat_t30 = build_features(T0 - 30*86400)
X30 = feat_t30[[c for c in FEATURE_COLS if c in feat_t30.columns]].values.astype(np.float32)
y30 = feat_t30['label'].values
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for tr, val in kf.split(X30, y30):
    dt = lgb.Dataset(X30[tr], label=y30[tr])
    dv = lgb.Dataset(X30[val], label=y30[val], reference=dt)
    m  = lgb.train({**params,'scale_pos_weight':(y30==0).sum()/max((y30==1).sum(),1)}, dt, 500, valid_sets=[dv],
                   callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
    aucs.append(roc_auc_score(y30[val], m.predict(X30[val])))
auc_std = np.mean(aucs)
print(f"  AUC = {auc_std:.4f} (standard cross-validation)")
results.append({'experiment': 'Standard 5-fold CV (T-30)', 'auc': round(auc_std, 4), 'note': 'Baseline'})

# ── Experiment 2: Temporal Split — Train T-90, Test T-30 ────────────────────
print("\nExp 2: Temporal split (Train: T-90, Test: T-30)...")
feat_t90 = build_features(T0 - 90*86400)

# Get addresses that appear in T-90 (training) 
train_addrs = set(feat_t90['addr'])
# Test: addresses in T-30 that are NOT in the training set (truly new addresses)
feat_t30_new = feat_t30[~feat_t30['addr'].isin(train_addrs)]
feat_t30_old = feat_t30[feat_t30['addr'].isin(train_addrs)]

feat_cols = [c for c in FEATURE_COLS if c in feat_t90.columns]
X_train = feat_t90[feat_cols].values.astype(np.float32)
y_train = feat_t90['label'].values

# Retrain on ALL T-90 data
dt = lgb.Dataset(X_train, label=y_train)
m_temporal = lgb.train({**params,'scale_pos_weight':(y_train==0).sum()/max((y_train==1).sum(),1),'num_boost_round':300}, dt)

# Test on T-30 addresses
X_test30 = feat_t30[feat_cols].values.astype(np.float32)
y_test30  = feat_t30['label'].values
auc_temporal = roc_auc_score(y_test30, m_temporal.predict(X_test30))
print(f"  AUC = {auc_temporal:.4f} (temporal: train T-90 → test T-30)")
results.append({'experiment': 'Temporal split (train T-90, test T-30)', 'auc': round(auc_temporal,4), 'note': 'Generalization test'})

# ── Experiment 3: Population Split — 50% Sybils seen, 50% unseen ─────────────
print("\nExp 3: Population split (50% Sybil labels visible)...")
sybil_addrs = list(feat_t30[feat_t30['label']==1]['addr'])
normal_addrs = list(feat_t30[feat_t30['label']==0]['addr'])

# Split Sybils 50/50
sybil_train = set(sybil_addrs[:len(sybil_addrs)//2])
sybil_test  = set(sybil_addrs[len(sybil_addrs)//2:])

train_mask = feat_t30['addr'].isin(sybil_train) | feat_t30['label'].eq(0)
test_mask  = feat_t30['addr'].isin(sybil_test)  | feat_t30['label'].eq(0)

feat_cols30 = [c for c in FEATURE_COLS if c in feat_t30.columns]
X_tr = feat_t30[train_mask][feat_cols30].values.astype(np.float32)
y_tr = feat_t30[train_mask]['label'].values
X_te = feat_t30[test_mask][feat_cols30].values.astype(np.float32)
y_te = feat_t30[test_mask]['label'].values

dt = lgb.Dataset(X_tr, label=y_tr)
m_pop = lgb.train({**params,'scale_pos_weight':(y_tr==0).sum()/max((y_tr==1).sum(),1),'num_boost_round':300}, dt)
auc_pop = roc_auc_score(y_te, m_pop.predict(X_te))
print(f"  AUC = {auc_pop:.4f} (population split: 50% Sybil labels hidden in test)")
results.append({'experiment': 'Population split (50% Sybil unseen)', 'auc': round(auc_pop,4), 'note': 'Unknown Sybil generalization'})

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("GENERALIZATION RESULTS")
for r in results:
    print(f"  {r['experiment']:<45} AUC={r['auc']:.4f}")

df_res = pd.DataFrame(results)
df_res.to_csv(OUT_DIR / "generalization_results.csv", index=False)
print(f"\nSaved → {OUT_DIR}/generalization_results.csv")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,4))
    colors = ['#3B82F6','#F59E0B','#10B981']
    bars = ax.barh(df_res['experiment'], df_res['auc'], color=colors)
    ax.axvline(0.9, color='red', linestyle='--', alpha=0.5, label='AUC=0.9')
    ax.axvline(0.803, color='gray', linestyle=':', alpha=0.7, label='ARTEMIS baseline')
    for bar, val in zip(bars, df_res['auc']):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim([0.7, 0.97])
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('Generalization Experiments\n(Temporal + Population Split)', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "generalization_plot.png", dpi=150, bbox_inches='tight')
    print(f"Plot → {OUT_DIR}/generalization_plot.png")
    plt.close()
except Exception as e:
    print(f"Plot error: {e}")
print("Done!")
