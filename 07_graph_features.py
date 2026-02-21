"""
07_graph_features.py — Graph-Based Feature Augmentation (LLMDetection alternative)

Builds a bipartite transaction graph from TXS2:
  Address ↔ NFT Collection (sale events)
  Address → Address (direct transfers)

Extracts per-address graph features:
  - Degree centrality (how many unique collections / counterparties)
  - PageRank (structural importance in the trading network)
  - Co-purchase diversity (Jaccard similarity with other Sybils)
  - Cycle involvement (addresses in short buy-sell cycles)

Tests whether graph features improve AUC beyond behavioral features alone.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset")
TXS_PATH     = DATA_DIR / "TXS2_1662_1861.csv"
TARGETS_PATH = DATA_DIR / "airdrop2_targets.txt"
FEATURES_PATH = DATA_DIR / "addresses_all_with_loaylty_blend_blur_label319.csv"
OUT_DIR = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")

T0     = 1700525735
CUTOFF = T0 - 30 * 86400   # T-30

print("Loading data...")
with open(TARGETS_PATH) as f:
    targets = set(line.strip().lower() for line in f if line.strip())

txs = pd.read_csv(TXS_PATH, usecols=['from','send','receive','timestamp','trade_price','event_type','contract_address'],
                  dtype={'timestamp':'Int64','trade_price':'float32'}, low_memory=False)
txs['ts'] = txs['timestamp'] // 1000
for c in ['from','send','receive']:
    txs[c] = txs[c].str.lower().fillna('')

ext = pd.read_csv(FEATURES_PATH)
ext['address'] = ext['address'].str.lower()
ext = ext[['address','blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio']].rename(columns={'address':'addr'})

df = txs[txs['ts'] < CUTOFF].copy()
sales = df[df['event_type'] == 'Sale'].copy()

print(f"  T-30 sales: {len(sales):,}")

# ── Graph Feature 1: Collection diversity per buyer/seller ────────────────────
buy_collections  = sales.groupby('send')['contract_address'].nunique().rename('graph_buy_diversity')
sell_collections = sales.groupby('receive')['contract_address'].nunique().rename('graph_sell_diversity')

# ── Graph Feature 2: Unique counterparties (buyers who bought from same seller) ─
# For each seller: how many unique buyers
seller_buyers = sales.groupby('receive')['send'].nunique().rename('graph_n_buyers')
# For each buyer: how many unique sellers
buyer_sellers = sales.groupby('send')['receive'].nunique().rename('graph_n_sellers')

# ── Graph Feature 3: Cycle detection (buy then sell same NFT = flip cycle) ───
# A cycle: addr appears as both buyer and seller of same contract_address
buy_set  = set(zip(sales['send'],    sales['contract_address']))
sell_set = set(zip(sales['receive'], sales['contract_address']))
cycle_addrs = set(a for (a,c) in buy_set if (a,c) in sell_set)
all_addrs = list(set(sales['send'].tolist() + sales['receive'].tolist()))
cycle_df = pd.DataFrame({'addr': all_addrs})
cycle_df['graph_cycle'] = cycle_df['addr'].isin(cycle_addrs).astype(int)

# ── Graph Feature 4: PageRank proxy (weighted degree) ────────────────────────
# Approximate via trade volume weighted by number of connections
buy_vol   = sales.groupby('send')['trade_price'].sum().rename('graph_buy_vol_total')
sell_vol  = sales.groupby('receive')['trade_price'].sum().rename('graph_sell_vol_total')
buy_cnt   = sales.groupby('send').size().rename('graph_buy_n')
sell_cnt  = sales.groupby('receive').size().rename('graph_sell_n')

# ── Graph Feature 5: NFT collection overlap with known Sybils ────────────────
# Jaccard: what fraction of this address's collections were also traded by Sybils
sybil_collections = set(sales[sales['send'].isin(targets)]['contract_address'].unique())
def jaccard_with_sybils(collections_str):
    return len(set(collections_str) & sybil_collections) / (len(set(collections_str)) + 1)

addr_collections = sales.groupby('send')['contract_address'].apply(list)
jaccard_s = addr_collections.apply(jaccard_with_sybils).rename('graph_sybil_collection_overlap')

# ── Merge all graph features ──────────────────────────────────────────────────
print("Merging graph features...")
graph_feat = cycle_df.set_index('addr')
graph_feat = graph_feat.join(buy_collections, how='left')
graph_feat = graph_feat.join(sell_collections, how='left')
graph_feat = graph_feat.join(seller_buyers, how='left')
graph_feat = graph_feat.join(buyer_sellers, how='left')
graph_feat = graph_feat.join(buy_vol, how='left')
graph_feat = graph_feat.join(sell_vol, how='left')
graph_feat = graph_feat.join(jaccard_s, how='left')
graph_feat = graph_feat.fillna(0).reset_index()
graph_feat.columns = ['addr'] + list(graph_feat.columns[1:])
print(f"  Graph features for {len(graph_feat):,} addresses")

# ── Build full feature matrix ─────────────────────────────────────────────────
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
feat = feat[feat['total_trade_count'] > 0]
feat = feat.merge(ext, on='addr', how='left')
feat = feat.merge(graph_feat, on='addr', how='left')
feat[['blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio',
      'graph_cycle','graph_buy_diversity','graph_sell_diversity','graph_n_buyers','graph_n_sellers',
      'graph_buy_vol_total','graph_sell_vol_total','graph_sybil_collection_overlap']] = \
    feat[['blend_in_count','blend_out_count','blend_net_value','LP_count','unique_interactions','ratio',
          'graph_cycle','graph_buy_diversity','graph_sell_diversity','graph_n_buyers','graph_n_sellers',
          'graph_buy_vol_total','graph_sell_vol_total','graph_sybil_collection_overlap']].fillna(0)
feat['label'] = feat['addr'].isin(targets).astype(int)
y = feat['label'].values
print(f"  {len(feat):,} addresses | {y.sum():,} Sybil ({y.mean()*100:.1f}%)")

BASE_COLS = ['buy_count','sell_count','tx_count','total_trade_count','buy_value','sell_value',
             'pnl_proxy','buy_collections','unique_interactions','sell_ratio','wallet_age_days',
             'days_since_last_buy','recent_activity','blend_in_count','blend_out_count',
             'blend_net_value','LP_count','ratio']
GRAPH_COLS = ['graph_cycle','graph_buy_diversity','graph_sell_diversity','graph_n_buyers',
              'graph_n_sellers','graph_buy_vol_total','graph_sell_vol_total','graph_sybil_collection_overlap']

def run_cv(X, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, val in kf.split(X, y):
        dt = lgb.Dataset(X[tr], label=y[tr])
        dv = lgb.Dataset(X[val], label=y[val], reference=dt)
        m  = lgb.train({'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
                        'scale_pos_weight':(y==0).sum()/max((y==1).sum(),1),'verbose':-1},
                       dt, 500, valid_sets=[dv],
                       callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
        aucs.append(roc_auc_score(y[val], m.predict(X[val])))
    return float(np.mean(aucs))

print("\nRunning experiments...")
base_feats  = [c for c in BASE_COLS  if c in feat.columns]
graph_feats = [c for c in GRAPH_COLS if c in feat.columns]

auc_base  = run_cv(feat[base_feats].values.astype(np.float32), y)
auc_graph = run_cv(feat[graph_feats].values.astype(np.float32), y)
auc_all   = run_cv(feat[base_feats+graph_feats].values.astype(np.float32), y)

print(f"\n{'='*50}")
print(f"Behavioral features only:   AUC = {auc_base:.4f}")
print(f"Graph features only:        AUC = {auc_graph:.4f}")
print(f"Behavioral + Graph (combined): AUC = {auc_all:.4f}")
print(f"Graph improvement:          +{auc_all-auc_base:.4f}")

results = pd.DataFrame([
    {'experiment': 'Behavioral only',       'auc': auc_base},
    {'experiment': 'Graph only',            'auc': auc_graph},
    {'experiment': 'Behavioral + Graph',    'auc': auc_all},
    {'experiment': 'ARTEMIS (post-hoc)',    'auc': 0.803},
])
results.to_csv(OUT_DIR / "graph_augmentation_results.csv", index=False)

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    colors = ['#3B82F6','#F59E0B','#10B981','#94A3B8']
    bars = ax.barh(results['experiment'], results['auc'], color=colors)
    ax.axvline(0.9, color='red', linestyle='--', alpha=0.5, label='AUC=0.9')
    for bar, val in zip(bars, results['auc']):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim([0.5, 0.97])
    ax.set_xlabel('AUC (5-fold CV, T-30)', fontsize=12)
    ax.set_title('Graph Feature Augmentation\nvs Behavioral-only and ARTEMIS', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "graph_augmentation_plot.png", dpi=150, bbox_inches='tight')
    print(f"Plot → {OUT_DIR}/graph_augmentation_plot.png")
    plt.close()
except Exception as e:
    print(f"Plot error: {e}")
print("Done!")
