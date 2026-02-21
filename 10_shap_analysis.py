"""10_shap_analysis.py — SHAP Explainability (T-30 model)"""
import pandas as pd, numpy as np, shap
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset")
OUT_DIR = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
T0, CUTOFF = 1700525735, 1700525735 - 30*86400

with open(DATA_DIR/"airdrop2_targets.txt") as f:
    targets = set(l.strip().lower() for l in f if l.strip())

print("Loading TXS2..."); import sys; sys.stdout.flush()
txs = pd.read_csv(DATA_DIR/"TXS2_1662_1861.csv",
    usecols=['from','send','receive','timestamp','trade_price','event_type','contract_address'],
    dtype={'timestamp':'Int64','trade_price':'float32'}, low_memory=False)
txs['ts'] = txs['timestamp']//1000
for c in ['from','send','receive']: txs[c]=txs[c].str.lower().fillna('')

ext = pd.read_csv(DATA_DIR/"addresses_all_with_loaylty_blend_blur_label319.csv")
ext['address']=ext['address'].str.lower()
ext=ext[['address','blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']].rename(columns={'address':'addr'})

df = txs[txs['ts']<CUTOFF]
buys = df[df['event_type']=='Sale'][['send','contract_address','trade_price','ts']].rename(columns={'send':'addr'})
sells= df[df['event_type']=='Sale'][['receive','contract_address','trade_price','ts']].rename(columns={'receive':'addr'})
all_a = pd.concat([buys[['addr']],sells[['addr']],df[['from']].rename(columns={'from':'addr'})]).drop_duplicates()
all_a = all_a[all_a['addr'].str.startswith('0x')]
buy_s = buys.groupby('addr').agg(buy_count=('addr','count'),buy_value=('trade_price','sum'),buy_collections=('contract_address','nunique'),buy_last_ts=('ts','max'),buy_first_ts=('ts','min')).reset_index()
sell_s= sells.groupby('addr').agg(sell_count=('addr','count'),sell_value=('trade_price','sum')).reset_index()
from_s= df.groupby('from').agg(tx_count=('ts','count'),first_tx_ts=('ts','min')).reset_index().rename(columns={'from':'addr'})
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

FEATURES = ['buy_count','sell_count','tx_count','total_trade_count','buy_value','sell_value',
            'pnl_proxy','buy_collections','unique_interactions','sell_ratio','wallet_age_days',
            'days_since_last_buy','recent_activity','blend_in_count','blend_out_count',
            'blend_net_value','LP_count','ratio']
FEAT_LABELS = {
    'buy_count':'Buy Count','sell_count':'Sell Count','tx_count':'Total Tx',
    'total_trade_count':'Total Trade Count','buy_value':'Buy Volume (ETH)',
    'sell_value':'Sell Volume (ETH)','pnl_proxy':'PnL Proxy','buy_collections':'NFT Diversity',
    'unique_interactions':'Unique Interactions','sell_ratio':'Sell Ratio',
    'wallet_age_days':'Wallet Age','days_since_last_buy':'Days Since Last Buy',
    'recent_activity':'Recent Activity','blend_in_count':'Blend Borrow',
    'blend_out_count':'Blend Repay','blend_net_value':'Blend Net Value',
    'LP_count':'LP Activity','ratio':'Volume Ratio'
}
fcols = [c for c in FEATURES if c in feat.columns]
X = feat[fcols].values.astype(np.float32); y = feat['label'].values
print(f"  {len(feat):,} addresses | {y.sum():,} Sybil"); sys.stdout.flush()

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
dt = lgb.Dataset(X_tr,label=y_tr); params={'objective':'binary','num_leaves':63,'learning_rate':0.05,
    'scale_pos_weight':(y==0).sum()/max((y==1).sum(),1),'verbose':-1}
model = lgb.train(params,dt,num_boost_round=300)
print(f"  Train AUC via SHAP model trained"); sys.stdout.flush()

print("Computing SHAP values..."); sys.stdout.flush()
explainer = shap.TreeExplainer(model)
# Sample 5000 for speed
idx = np.random.choice(len(X_te), min(5000,len(X_te)), replace=False)
shap_values = explainer.shap_values(X_te[idx])

# Summary
shap_abs = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({'feature':fcols,'shap_importance':shap_abs,'label':[FEAT_LABELS.get(c,c) for c in fcols]})
shap_df = shap_df.sort_values('shap_importance',ascending=False)
print("\nSHAP Feature Importance (mean |SHAP|):")
for _,r in shap_df.iterrows():
    print(f"  {r['label']:<30} {r['shap_importance']:.4f}")
shap_df.to_csv(OUT_DIR/"shap_importance.csv",index=False)

try:
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    top=shap_df.head(12)
    ax.barh(top['label'],top['shap_importance'],color='#3B82F6',alpha=0.85)
    ax.invert_yaxis(); ax.set_xlabel('Mean |SHAP Value|',fontsize=12)
    ax.set_title('SHAP Feature Importance\n(T-30 Pre-Airdrop Model, n=5000 test samples)',fontsize=13,fontweight='bold')
    ax.grid(axis='x',alpha=0.3); plt.tight_layout()
    fig.savefig(OUT_DIR/"shap_plot.png",dpi=150,bbox_inches='tight')
    print(f"Plot → {OUT_DIR}/shap_plot.png"); plt.close()
    # SHAP beeswarm
    fig2,ax2=plt.subplots(figsize=(10,7))
    shap.summary_plot(shap_values,X_te[idx],feature_names=[FEAT_LABELS.get(c,c) for c in fcols],show=False,max_display=12)
    plt.tight_layout()
    fig2.savefig(OUT_DIR/"shap_beeswarm.png",dpi=150,bbox_inches='tight')
    print(f"Beeswarm → {OUT_DIR}/shap_beeswarm.png"); plt.close('all')
except Exception as e: print(f"Plot err: {e}")
print("Done!")
