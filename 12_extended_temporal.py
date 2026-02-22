"""12_extended_temporal.py — Extended Detection Limit: T-120, T-150, T-180"""
import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset")
OUT_DIR  = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
T0 = 1700525735

WINDOWS = {
    "T-30":  T0 - 30*86400,   # Already done, baseline
    "T-60":  T0 - 60*86400,
    "T-90":  T0 - 90*86400,
    "T-120": T0 - 120*86400,  # NEW
    "T-150": T0 - 150*86400,  # NEW
    "T-180": T0 - 180*86400,  # NEW (May 25, 2023)
}

def ts_label(ts): return datetime.fromtimestamp(ts,tz=timezone.utc).strftime("%Y-%m-%d")

_flags = pd.read_csv(DATA_DIR / "airdrop_targets_behavior_flags.csv")
targets = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
del _flags

print("Loading TXS2 (reuse for all windows)...")
import sys; sys.stdout.flush()
txs = pd.read_csv(DATA_DIR/"TXS2_1662_1861.csv",
    usecols=['from','send','receive','timestamp','trade_price','event_type','contract_address'],
    dtype={'timestamp':'Int64','trade_price':'float32'}, low_memory=False)
txs['ts']=txs['timestamp']//1000
for c in ['from','send','receive']: txs[c]=txs[c].str.lower().fillna('')
ext=pd.read_csv(DATA_DIR/"addresses_all_with_loaylty_blend_blur_label319.csv")
ext['address']=ext['address'].str.lower()
ext=ext[['address','blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']].rename(columns={'address':'addr'})
print(f"  TXS2 range: {ts_label(txs.ts.min())} → {ts_label(txs.ts.max())}")
sys.stdout.flush()

results=[]
for wname, cutoff in WINDOWS.items():
    df=txs[txs['ts']<cutoff]
    if len(df)<1000:
        print(f"{wname}: too few transactions"); continue
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
    feat['wallet_age_days']=(cutoff-feat['first_tx_ts'].clip(lower=0))/86400
    feat['days_since_last_buy']=(cutoff-feat['buy_last_ts'].clip(lower=0))/86400
    feat['recent_activity']=(feat['buy_last_ts']>(cutoff-30*86400)).astype(int)
    feat=feat[feat['total_trade_count']>0].merge(ext,on='addr',how='left')
    for c in ['blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']:
        feat[c]=feat[c].fillna(0)
    feat=feat[feat['addr'].isin(all_recipients)].copy()   # restrict to 53K airdrop recipients
    feat['label']=feat['addr'].isin(targets).astype(int)
    FCOLS=['buy_count','sell_count','tx_count','total_trade_count','buy_value','sell_value','pnl_proxy',
           'buy_collections','unique_interactions','sell_ratio','wallet_age_days','days_since_last_buy',
           'recent_activity','blend_in_count','blend_out_count','blend_net_value','LP_count','ratio']
    fcols=[c for c in FCOLS if c in feat.columns]
    X=feat[fcols].values.astype(np.float32); y=feat['label'].values
    kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    aucs=[]
    for tr,val in kf.split(X,y):
        dt=lgb.Dataset(X[tr],label=y[tr]); dv=lgb.Dataset(X[val],label=y[val],reference=dt)
        m=lgb.train({'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
                     'scale_pos_weight':(y==0).sum()/max((y==1).sum(),1),'verbose':-1},
                    dt,500,valid_sets=[dv],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
        aucs.append(roc_auc_score(y[val],m.predict(X[val])))
    auc=np.mean(aucs)
    days=(T0-cutoff)//86400
    results.append({'window':wname,'cutoff':ts_label(cutoff),'days_before_T0':days,
                    'n_addresses':len(feat),'n_sybil':int(y.sum()),'auc':round(auc,4)})
    print(f"  {wname:6s} ({ts_label(cutoff)}) | {len(feat):,} addr | AUC={auc:.4f}")
    sys.stdout.flush()

df_res=pd.DataFrame(results)
# Merge with existing temporal ablation results
try:
    existing=pd.read_csv(OUT_DIR/"temporal_ablation_results.csv")
    combined=pd.concat([existing[~existing['window'].isin(df_res['window'])],df_res]).sort_values('days_before_T0')
    combined.to_csv(OUT_DIR/"temporal_ablation_extended.csv",index=False)
    print(f"\nExtended temporal ablation:")
    print(combined[['window','cutoff','auc']].to_string(index=False))
except: df_res.to_csv(OUT_DIR/"temporal_ablation_extended.csv",index=False)

try:
    import matplotlib.pyplot as plt
    try: combined
    except: combined=df_res
    fig,ax=plt.subplots(figsize=(11,5))
    ax.plot(combined['days_before_T0'],combined['auc'],'o-',color='#3B82F6',linewidth=2.5,markersize=9)
    ax.axhline(0.9,color='red',linestyle='--',alpha=0.5,label='AUC=0.9')
    ax.axhline(0.803,color='gray',linestyle=':',alpha=0.7,label='ARTEMIS baseline (0.803)')
    ax.set_xlabel('Days Before Airdrop (T0)',fontsize=12); ax.set_ylabel('AUC (5-fold CV)',fontsize=12)
    ax.set_title('Detection AUC vs Observation Window\n(Extended: T-180 to T0)',fontsize=13,fontweight='bold')
    ax.invert_xaxis(); ax.set_ylim([0.75,0.95]); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    for _,r in combined.iterrows():
        ax.annotate(f"{r['window']}\n{r['auc']:.3f}",(r['days_before_T0'],r['auc']),
                    textcoords="offset points",xytext=(0,10),ha='center',fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR/"temporal_ablation_extended_plot.png",dpi=150,bbox_inches='tight')
    print(f"Plot → {OUT_DIR}/temporal_ablation_extended_plot.png"); plt.close()
except Exception as e: print(f"Plot err: {e}")
print("Done!")
