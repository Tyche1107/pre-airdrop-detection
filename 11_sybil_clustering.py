"""11_sybil_clustering.py — Sybil Strategy Clustering (T-30 features)"""
import pandas as pd, numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset")
OUT_DIR = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
T0, CUTOFF = 1700525735, 1700525735-30*86400

_flags = pd.read_csv(DATA_DIR / "airdrop_targets_behavior_flags.csv")
targets = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
all_recipients = set(_flags['address'].str.lower())   # 53K airdrop recipients
del _flags

print("Loading TXS2...")
import sys; sys.stdout.flush()
txs = pd.read_csv(DATA_DIR/"TXS2_1662_1861.csv",
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
feat=feat[feat['addr'].isin(all_recipients)].copy()   # restrict to 53K airdrop recipients
feat['label']=feat['addr'].isin(targets).astype(int)

# Focus on confirmed Sybils only
sybils = feat[feat['label']==1].copy()
CLUSTER_FEATS=['buy_count','sell_count','buy_value','sell_value','buy_collections',
               'sell_ratio','wallet_age_days','pnl_proxy','blend_in_count','unique_interactions']
X=sybils[[c for c in CLUSTER_FEATS if c in sybils.columns]].values.astype(np.float32)
scaler=StandardScaler(); Xs=scaler.fit_transform(X)
print(f"  {len(sybils):,} Sybil addresses to cluster"); sys.stdout.flush()

# Find optimal K
sil_scores=[]
for k in range(2,8):
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    labels=km.fit_predict(Xs)
    if len(np.unique(labels)) < 2:
        print(f"  K={k} skipped (only 1 unique cluster label)"); sys.stdout.flush()
        continue
    try:
        # Use min(5000, len(Xs)-1) to avoid issues with small datasets
        n_sample = min(5000, len(Xs)) if len(Xs) > 5000 else None
        score = silhouette_score(Xs, labels, sample_size=n_sample, random_state=42)
        sil_scores.append((k, score))
        print(f"  K={k} silhouette={sil_scores[-1][1]:.4f}"); sys.stdout.flush()
    except Exception as e:
        print(f"  K={k} silhouette error: {e}"); sys.stdout.flush()

if not sil_scores:
    print("No valid silhouette scores computed, defaulting to K=2")
    sil_scores = [(2, 0.0)]

best_k=max(sil_scores,key=lambda x:x[1])[0]
print(f"\nBest K={best_k}")
km=KMeans(n_clusters=best_k,random_state=42,n_init=10)
sybils['cluster']=km.fit_predict(Xs)

# Profile each cluster
print("\nCluster Profiles:")
feat_names={'buy_count':'Buys','sell_count':'Sells','buy_value':'Buy Vol ETH',
            'buy_collections':'Collections','sell_ratio':'Sell%','wallet_age_days':'Age(days)',
            'blend_in_count':'Blend Borrows'}
for cl in range(best_k):
    sub=sybils[sybils['cluster']==cl]
    medians={k:sub[k].median() for k in CLUSTER_FEATS if k in sub.columns}
    label=f"Cluster {cl} (n={len(sub):,})"
    print(f"\n  {label}")
    for k,v in medians.items():
        print(f"    {feat_names.get(k,k):<20} {v:.1f}")

# Strategy labels based on cluster profiles
cluster_profiles=sybils.groupby('cluster')[CLUSTER_FEATS].median()
cluster_profiles['size']=sybils.groupby('cluster').size()
cluster_profiles.to_csv(OUT_DIR/"sybil_cluster_profiles.csv")
print(f"\nSaved → {OUT_DIR}/sybil_cluster_profiles.csv")

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    # K selection
    ks=[s[0] for s in sil_scores]; sils=[s[1] for s in sil_scores]
    axes[0].plot(ks,sils,'o-',color='#3B82F6',linewidth=2,markersize=8)
    axes[0].axvline(best_k,color='red',linestyle='--',alpha=0.7,label=f'Best K={best_k}')
    axes[0].set_xlabel('Number of Clusters K'); axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Optimal K Selection\n(Sybil Strategy Clustering)'); axes[0].legend(); axes[0].grid(alpha=0.3)
    # Cluster sizes
    sizes=sybils.groupby('cluster').size()
    axes[1].bar(range(best_k),[sizes.get(i,0) for i in range(best_k)],
                color=[f'C{i}' for i in range(best_k)],edgecolor='white')
    axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('Number of Sybils')
    axes[1].set_title(f'Sybil Cluster Sizes\n(K={best_k} strategies identified)')
    for i,(x,h) in enumerate(zip(range(best_k),[sizes.get(i,0) for i in range(best_k)])):
        axes[1].text(x,h+10,str(h),ha='center',fontsize=10,fontweight='bold')
    axes[1].grid(axis='y',alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR/"sybil_clustering_plot.png",dpi=150,bbox_inches='tight')
    print(f"Plot → {OUT_DIR}/sybil_clustering_plot.png"); plt.close()
except Exception as e: print(f"Plot err: {e}")
print("Done!")
