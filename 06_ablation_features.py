"""
06_ablation_features.py — Feature Group Ablation Study

测试各特征组的独立贡献，回答"哪类行为特征最有判别力"。

Feature Groups:
  A: Trading Activity     (buy_count, sell_count, tx_count, total_trade_count)
  B: Trading Volume       (buy_value, sell_value, pnl_proxy)
  C: Diversity            (buy_collections, unique_interactions)
  D: Behavioral Patterns  (sell_ratio, wallet_age_days, days_since_last_buy, recent_activity)
  E: DeFi/Blend Protocol  (blend_in_count, blend_out_count, blend_net_value, LP_count)
  ALL: All features combined
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset")
TXS_PATH  = DATA_DIR / "TXS2_1662_1861.csv"
FEATURES_PATH = DATA_DIR / "addresses_all_with_loaylty_blend_blur_label319.csv"
OUT_DIR = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
OUT_DIR.mkdir(exist_ok=True)

T0 = 1700525735
CUTOFF = T0 - 30 * 86400   # T-30 (main result window)

FEATURE_GROUPS = {
    'A_Activity':   ['buy_count', 'sell_count', 'tx_count', 'total_trade_count'],
    'B_Volume':     ['buy_value', 'sell_value', 'pnl_proxy'],
    'C_Diversity':  ['buy_collections', 'unique_interactions'],
    'D_Behavioral': ['sell_ratio', 'wallet_age_days', 'days_since_last_buy', 'recent_activity'],
    'E_DeFi':       ['blend_in_count', 'blend_out_count', 'blend_net_value', 'LP_count'],
    'ALL':          None,  # all features
}

ALL_FEATURES = [
    'buy_count','sell_count','tx_count','total_trade_count',
    'buy_value','sell_value','pnl_proxy',
    'buy_collections','unique_interactions',
    'sell_ratio','wallet_age_days','days_since_last_buy','recent_activity',
    'blend_in_count','blend_out_count','blend_net_value','LP_count',
    'ratio'
]

print("Loading data...")
_flags = pd.read_csv(DATA_DIR / "airdrop_targets_behavior_flags.csv")
targets = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
del _flags

txs = pd.read_csv(TXS_PATH, usecols=['from','send','receive','timestamp','trade_price','event_type','contract_address'],
                  dtype={'timestamp':'Int64','trade_price':'float32'}, low_memory=False)
txs['ts'] = txs['timestamp'] // 1000
for col in ['from','send','receive']:
    txs[col] = txs[col].str.lower().fillna('')

ext = pd.read_csv(FEATURES_PATH)
ext['address'] = ext['address'].str.lower()
ext = ext[['address','blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']].rename(columns={'address':'addr'})

print("Building T-30 feature matrix...")
df = txs[txs['ts'] < CUTOFF].copy()
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
feat['wallet_age_days']   = (CUTOFF - feat['first_tx_ts'].clip(lower=0)) / 86400
feat['days_since_last_buy'] = (CUTOFF - feat['buy_last_ts'].clip(lower=0)) / 86400
feat['recent_activity']   = (feat['buy_last_ts'] > (CUTOFF - 30*86400)).astype(int)
feat = feat[feat['total_trade_count'] > 0].merge(ext, on='addr', how='left')
feat[['blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']] = \
    feat[['blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']].fillna(0)
feat['label'] = feat['addr'].isin(targets).astype(int)
y = feat['label'].values
print(f"  {len(feat):,} addresses | {y.sum():,} Sybil ({y.mean()*100:.1f}%)\n")

def run_cv(X, y, name):
    kf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, val in kf.split(X, y):
        dt = lgb.Dataset(X[tr], label=y[tr])
        dv = lgb.Dataset(X[val], label=y[val], reference=dt)
        m  = lgb.train({'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':31,
                        'scale_pos_weight':(y==0).sum()/max((y==1).sum(),1),'verbose':-1},
                       dt, 300, valid_sets=[dv],
                       callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(-1)])
        aucs.append(roc_auc_score(y[val], m.predict(X[val])))
    return float(np.mean(aucs))

results = []
for gname, cols in FEATURE_GROUPS.items():
    use_cols = [c for c in (ALL_FEATURES if cols is None else cols) if c in feat.columns]
    X = feat[use_cols].values.astype(np.float32)
    auc = run_cv(X, y, gname)
    results.append({'group': gname, 'features': ', '.join(use_cols), 'n_features': len(use_cols), 'auc': round(auc,4)})
    print(f"  {gname:<18} {len(use_cols):>2} feats  AUC={auc:.4f}")

print("\n=== ABLATION RESULTS (T-30) ===")
df_res = pd.DataFrame(results).sort_values('auc', ascending=False)
print(df_res[['group','n_features','auc']].to_string(index=False))
df_res.to_csv(OUT_DIR / "ablation_results.csv", index=False)

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    colors = ['#10B981' if r['group']=='ALL' else '#3B82F6' for _,r in df_res.iterrows()]
    bars = ax.barh(df_res['group'], df_res['auc'], color=colors, edgecolor='white')
    ax.axvline(0.9, color='red', linestyle='--', alpha=0.5, label='AUC=0.9')
    ax.axvline(0.803, color='gray', linestyle=':', alpha=0.7, label='ARTEMIS baseline')
    ax.set_xlabel('AUC (5-fold CV, T-30)', fontsize=12)
    ax.set_title('Feature Group Ablation Study\n(Pre-airdrop Sybil Detection, T-30)', fontsize=13, fontweight='bold')
    for bar, row in zip(bars, df_res.itertuples()):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2, f'{row.auc:.4f}', va='center', fontsize=10)
    ax.set_xlim([0.5, 0.96])
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ablation_plot.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved → {OUT_DIR}/ablation_plot.png")
    plt.close()
except Exception as e:
    print(f"Plot error: {e}")
print("Done!")
