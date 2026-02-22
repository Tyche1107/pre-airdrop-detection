"""13_adversarial_robustness.py — Adversarial Robustness Analysis

If Sybils know they're being detected, how would they evade?
Simulation: Perturb Sybil features (reduce diversity, mimic normal users)
and measure AUC degradation. Shows detection upper/lower bounds.
"""
import pandas as pd, numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset")
OUT_DIR  = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
T0, CUTOFF = 1700525735, 1700525735-30*86400

_flags = pd.read_csv(DATA_DIR / "airdrop_targets_behavior_flags.csv")
targets = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
del _flags

print("Loading data...")
import sys; sys.stdout.flush()
txs=pd.read_csv(DATA_DIR/"TXS2_1662_1861.csv",
    usecols=['from','send','receive','timestamp','trade_price','event_type','contract_address'],
    dtype={'timestamp':'Int64','trade_price':'float32'}, low_memory=False)
txs['ts']=txs['timestamp']//1000
for c in ['from','send','receive']: txs[c]=txs[c].str.lower().fillna('')
ext=pd.read_csv(DATA_DIR/"addresses_all_with_loaylty_blend_blur_label319.csv")
ext['address']=ext['address'].str.lower()
ext=ext[['address','blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']].rename(columns={'address':'addr'})

df=txs[txs['ts']<CUTOFF]
buys=df[df['event_type']=='Sale'][['send','contract_address','trade_price','ts']].rename(columns={'send':'addr'})
sells=df[df['event_type']=='Sale'][['receive','contract_address','trade_price','ts']].rename(columns={'receive':'addr'})
all_a=pd.concat([buys[['addr']],sells[['addr']],df[['from']].rename(columns={'from':'addr'})]).drop_duplicates()
all_a=all_a[all_a['addr'].str.startswith('0x')]
buy_s=buys.groupby('addr').agg(buy_count=('addr','count'),buy_value=('trade_price','sum'),buy_collections=('contract_address','nunique'),buy_last_ts=('ts','max'),buy_first_ts=('ts','min')).reset_index()
sell_s=sells.groupby('addr').agg(sell_count=('addr','count'),sell_value=('trade_price','sum')).reset_index()
from_s=df.groupby('from').agg(tx_count=('ts','count'),first_tx_ts=('ts','min')).reset_index().rename(columns={'from':'addr'})
feat=all_a.merge(buy_s,on='addr',how='left').merge(sell_s,on='addr',how='left').merge(from_s,on='addr',how='left').fillna(0)
feat['total_trade_count']=feat['buy_count']+feat['sell_count']
feat['sell_ratio']=feat['sell_count']/(feat['total_trade_count']+1e-6)
feat['pnl_proxy']=feat['sell_value']-feat['buy_value']
feat['wallet_age_days']=(CUTOFF-feat['first_tx_ts'].clip(lower=0))/86400
feat['days_since_last_buy']=(CUTOFF-feat['buy_last_ts'].clip(lower=0))/86400
feat['recent_activity']=(feat['buy_last_ts']>(CUTOFF-30*86400)).astype(int)
feat=feat[feat['total_trade_count']>0].merge(ext,on='addr',how='left')
for c in ['blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']:
    feat[c]=feat[c].fillna(0)
feat['label']=feat['addr'].isin(targets).astype(int)
print(f"  {len(feat):,} addresses | {feat['label'].sum():,} Sybil"); sys.stdout.flush()

FCOLS=['buy_count','sell_count','tx_count','total_trade_count','buy_value','sell_value','pnl_proxy',
       'buy_collections','unique_interactions','sell_ratio','wallet_age_days','days_since_last_buy',
       'recent_activity','blend_in_count','blend_out_count','blend_net_value','LP_count','ratio']
fcols=[c for c in FCOLS if c in feat.columns]

# Train baseline model on clean data
kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
X=feat[fcols].values.astype(np.float32); y=feat['label'].values

def eval_auc(X_test, y_test, models):
    return float(np.mean([roc_auc_score(y_test[val], m.predict(X_test[val]))
                          for m,(_,val) in zip(models, kf.split(X_test,y_test))]))

# Train base models
base_models=[]
for tr,val in kf.split(X,y):
    dt=lgb.Dataset(X[tr],label=y[tr]); dv=lgb.Dataset(X[val],label=y[val],reference=dt)
    m=lgb.train({'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
                 'scale_pos_weight':(y==0).sum()/max((y==1).sum(),1),'verbose':-1},
                dt,500,valid_sets=[dv],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
    base_models.append(m)
base_auc=float(np.mean([roc_auc_score(y[val],base_models[i].predict(X[val]))
                         for i,(_,val) in enumerate(kf.split(X,y))]))
print(f"\nBaseline AUC (clean data): {base_auc:.4f}")

results=[{'scenario':'Baseline (no evasion)','auc':round(base_auc,4),'description':'Normal Sybil behavior'}]

# Evasion scenarios
sybil_idx = np.where(y==1)[0]
normal_idx = np.where(y==0)[0]
col_idx = fcols.index('buy_collections') if 'buy_collections' in fcols else None
int_idx = fcols.index('unique_interactions') if 'unique_interactions' in fcols else None

scenarios = [
    ("Reduce NFT diversity -50%", 'buy_collections', 0.5),
    ("Reduce NFT diversity -80%", 'buy_collections', 0.2),
    ("Reduce unique interactions -50%", 'unique_interactions', 0.5),
    ("Mimic normal user diversity", 'buy_collections', None),  # set to normal median
]

for name, feat_name, scale in scenarios:
    X_adv = X.copy()
    fi = fcols.index(feat_name) if feat_name in fcols else None
    if fi is None: continue
    if scale is not None:
        X_adv[sybil_idx, fi] *= scale
    else:
        normal_median = np.median(X[normal_idx, fi])
        X_adv[sybil_idx, fi] = normal_median

    fold_aucs=[]
    for i,(_,val) in enumerate(kf.split(X_adv,y)):
        fold_aucs.append(roc_auc_score(y[val], base_models[i].predict(X_adv[val])))
    auc=float(np.mean(fold_aucs))
    delta=auc-base_auc
    print(f"  {name:<45} AUC={auc:.4f} ({delta:+.4f})")
    results.append({'scenario':name,'auc':round(auc,4),'description':f'{feat_name} reduced'})
    sys.stdout.flush()

# Evasion cost: how much does a Sybil need to reduce activity to avoid detection?
print("\nEvasion cost analysis:")
print("(How much must Sybil reduce collection diversity to drop AUC below 0.85?)")
for scale in [0.9,0.7,0.5,0.3,0.1,0.0]:
    X_adv=X.copy()
    if col_idx: X_adv[sybil_idx,col_idx]*=scale
    fold_aucs=[roc_auc_score(y[val],base_models[i].predict(X_adv[val]))
               for i,(_,val) in enumerate(kf.split(X_adv,y))]
    auc=float(np.mean(fold_aucs))
    print(f"  Diversity × {scale:.1f}  → AUC={auc:.4f}")
    sys.stdout.flush()
    if auc < 0.85: print(f"  ↑ AUC drops below 0.85 at {scale:.1f}x diversity")

df_res=pd.DataFrame(results)
df_res.to_csv(OUT_DIR/"adversarial_results.csv",index=False)
print(f"\nSaved → {OUT_DIR}/adversarial_results.csv")

try:
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    colors=['#10B981' if r['scenario']=='Baseline (no evasion)' else '#EF4444' for _,r in df_res.iterrows()]
    bars=ax.barh(df_res['scenario'],df_res['auc'],color=colors,alpha=0.85)
    ax.axvline(0.85,color='orange',linestyle='--',alpha=0.7,label='AUC=0.85 threshold')
    ax.axvline(base_auc,color='green',linestyle='-',alpha=0.5,label=f'Baseline {base_auc:.3f}')
    for bar,val in zip(bars,df_res['auc']):
        ax.text(bar.get_width()+0.001,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=10)
    ax.set_xlim([0.5,0.97]); ax.set_xlabel('AUC',fontsize=12)
    ax.set_title('Adversarial Robustness\n(Sybil evasion simulation)',fontsize=13,fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='x',alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_DIR/"adversarial_plot.png",dpi=150,bbox_inches='tight')
    print(f"Plot → {OUT_DIR}/adversarial_plot.png"); plt.close()
except Exception as e: print(f"Plot err: {e}")
print("Done!")
