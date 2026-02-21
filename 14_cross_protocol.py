"""14_cross_protocol.py — Cross-Protocol Generalization
Train on Blur (NFT), zero-shot test on Hop (bridge)
Using only protocol-agnostic behavioral features.
"""
import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings; warnings.filterwarnings('ignore')
import sys

DATA_DIR  = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset")
HOP_DIR   = Path("/Users/adelinewen/Desktop/dataset/hop")
OUT_DIR   = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")

# ── Timing ──────────────────────────────────────────────────────────────────
T0_BLUR  = 1700525735          # 2023-11-21  Blur Season 2 airdrop
T0_HOP   = 1649030400          # 2022-04-04  Hop Protocol airdrop
CUT_BLUR = T0_BLUR - 30*86400  # T-30 for Blur (our main window)

# ── 1. Build Blur dataset with GENERIC features only ────────────────────────
print("=== Building Blur generic features (T-30) ===")
with open(DATA_DIR/"airdrop2_targets.txt") as f:
    blur_sybils = set(l.strip().lower() for l in f if l.strip())

txs = pd.read_csv(DATA_DIR/"TXS2_1662_1861.csv",
    usecols=['from','send','receive','timestamp','trade_price','event_type'],
    dtype={'timestamp':'Int64','trade_price':'float32'}, low_memory=False)
txs['ts'] = txs['timestamp']//1000
for c in ['from','send','receive']: txs[c]=txs[c].str.lower().fillna('')

df = txs[txs['ts'] < CUT_BLUR]
buys  = df[df['event_type']=='Sale'][['send','trade_price','ts']].rename(columns={'send':'addr'})
sells = df[df['event_type']=='Sale'][['receive','trade_price','ts']].rename(columns={'receive':'addr'})
all_a = pd.concat([buys[['addr']],sells[['addr']],df[['from']].rename(columns={'from':'addr'})]).drop_duplicates()
all_a = all_a[all_a['addr'].str.startswith('0x')]

buy_s  = buys.groupby('addr').agg(buy_count=('addr','count'), buy_value=('trade_price','sum'), buy_first=('ts','min')).reset_index()
sell_s = sells.groupby('addr').agg(sell_count=('addr','count'), sell_value=('trade_price','sum')).reset_index()
from_s = df.groupby('from').agg(tx_count=('ts','count'), first_ts=('ts','min')).reset_index().rename(columns={'from':'addr'})

feat = all_a.merge(buy_s,on='addr',how='left').merge(sell_s,on='addr',how='left').merge(from_s,on='addr',how='left').fillna(0)
feat['total_trades'] = feat['buy_count'] + feat['sell_count']
feat['total_volume'] = feat['buy_value'] + feat['sell_value']
feat['sell_ratio']   = feat['sell_count'] / (feat['total_trades']+1e-6)
feat['wallet_age']   = (CUT_BLUR - feat['first_ts'].clip(lower=0)) / 86400
feat['avg_vol_per_tx'] = feat['total_volume'] / (feat['total_trades']+1e-6)
feat = feat[feat['total_trades']>0]
feat['label'] = feat['addr'].isin(blur_sybils).astype(int)

GENERIC = ['tx_count','total_trades','total_volume','sell_ratio','wallet_age','avg_vol_per_tx']
fcols   = [c for c in GENERIC if c in feat.columns]
X_blur  = feat[fcols].values.astype(np.float32)
y_blur  = feat['label'].values
print(f"  Blur: {len(feat):,} addresses | {y_blur.sum():,} Sybil ({y_blur.mean()*100:.1f}%)")
sys.stdout.flush()

# ── 2. Build Hop dataset ────────────────────────────────────────────────────
print("\n=== Building Hop dataset ===")
hop_meta  = pd.read_csv(HOP_DIR/"metadata.csv")
hop_meta['address'] = hop_meta['address'].str.lower()
hop_meta.rename(columns={'totalTxs':'tx_count','totalVolume':'total_volume'}, inplace=True)

hop_ts = pd.read_csv(HOP_DIR/"timestamps.csv")
hop_ts['address'] = hop_ts['address'].str.lower()
hop_sybils = pd.read_csv(HOP_DIR/"sybil_addresses.csv")
hop_sybils['address'] = hop_sybils['address'].str.lower()
hop_sybil_set = set(hop_sybils['address'])

hop = hop_meta.merge(hop_ts, on='address', how='left')
hop['wallet_age']     = (T0_HOP - hop['first_ts'].clip(lower=0)) / 86400
hop['wallet_age']     = hop['wallet_age'].fillna(hop['wallet_age'].median())
hop['total_volume']   = hop['total_volume'].abs()  # sanity
hop['sell_ratio']     = 0.5  # Hop doesn't have buy/sell distinction (bridge only)
hop['avg_vol_per_tx'] = hop['total_volume'] / (hop['tx_count']+1e-6)
hop['total_trades']   = hop['tx_count']
hop['label']          = hop['address'].isin(hop_sybil_set).astype(int)

# Only keep addresses with actual activity
hop = hop[(hop['tx_count']>0) & (hop['total_volume']>0)]
X_hop = hop[fcols].values.astype(np.float32)
y_hop = hop['label'].values
print(f"  Hop:  {len(hop):,} addresses | {y_hop.sum():,} Sybil ({y_hop.mean()*100:.1f}%)")
sys.stdout.flush()

# ── 3. Train on Blur (5-fold CV), evaluate in-domain ───────────────────────
print("\n=== Train on Blur (in-domain) ===")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
blur_models, blur_aucs = [], []
for tr, val in kf.split(X_blur, y_blur):
    dt = lgb.Dataset(X_blur[tr], label=y_blur[tr])
    dv = lgb.Dataset(X_blur[val], label=y_blur[val], reference=dt)
    m  = lgb.train(
        {'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
         'scale_pos_weight':(y_blur==0).sum()/max((y_blur==1).sum(),1),'verbose':-1},
        dt, 500, valid_sets=[dv],
        callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    blur_models.append(m)
    blur_aucs.append(roc_auc_score(y_blur[val], m.predict(X_blur[val])))

blur_auc_mean = np.mean(blur_aucs)
print(f"  Blur in-domain AUC: {blur_auc_mean:.4f} ± {np.std(blur_aucs):.4f}")
sys.stdout.flush()

# Full Blur model (trained on all data) for zero-shot Hop
full_model = lgb.train(
    {'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
     'scale_pos_weight':(y_blur==0).sum()/max((y_blur==1).sum(),1),'verbose':-1},
    lgb.Dataset(X_blur, label=y_blur), 300)

# ── 4. Zero-shot: Blur model → Hop ─────────────────────────────────────────
print("\n=== Zero-shot: Blur → Hop ===")
hop_preds = full_model.predict(X_hop)
hop_auc   = roc_auc_score(y_hop, hop_preds)
hop_ap    = average_precision_score(y_hop, hop_preds)
print(f"  Zero-shot AUC: {hop_auc:.4f}  AP: {hop_ap:.4f}")
sys.stdout.flush()

# ── 5. Hop-only baseline (train on Hop, 5-fold) for comparison ─────────────
print("\n=== Hop-only baseline (same protocol) ===")
hop_aucs = []
for tr, val in kf.split(X_hop, y_hop):
    dt = lgb.Dataset(X_hop[tr], label=y_hop[tr])
    dv = lgb.Dataset(X_hop[val], label=y_hop[val], reference=dt)
    m  = lgb.train(
        {'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
         'scale_pos_weight':(y_hop==0).sum()/max((y_hop==1).sum(),1),'verbose':-1},
        dt, 500, valid_sets=[dv],
        callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    hop_aucs.append(roc_auc_score(y_hop[val], m.predict(X_hop[val])))
hop_baseline = np.mean(hop_aucs)
print(f"  Hop in-domain AUC: {hop_baseline:.4f} ± {np.std(hop_aucs):.4f}")
sys.stdout.flush()

# ── 6. Fine-tuned: 10% Hop labeled → rest test ─────────────────────────────
print("\n=== Fine-tuning: 10% Hop labels ===")
idx_sybil = np.where(y_hop==1)[0]
idx_norm  = np.where(y_hop==0)[0]
rng = np.random.RandomState(42)
ft_sybil = rng.choice(idx_sybil, int(len(idx_sybil)*0.1), replace=False)
ft_norm  = rng.choice(idx_norm,  int(len(idx_norm)*0.1),  replace=False)
ft_idx   = np.concatenate([ft_sybil, ft_norm])
test_idx = np.setdiff1d(np.arange(len(y_hop)), ft_idx)

ft_model = lgb.train(
    {'objective':'binary','learning_rate':0.05,'num_leaves':31,
     'scale_pos_weight':(y_hop[ft_idx]==0).sum()/max((y_hop[ft_idx]==1).sum(),1),'verbose':-1},
    lgb.Dataset(X_hop[ft_idx], label=y_hop[ft_idx]), 100)
ft_auc = roc_auc_score(y_hop[test_idx], ft_model.predict(X_hop[test_idx]))
print(f"  Fine-tuned AUC (10% labels): {ft_auc:.4f}")
sys.stdout.flush()

# ── 7. Save results ─────────────────────────────────────────────────────────
results = pd.DataFrame([
    {'experiment': 'Blur in-domain (generic features)', 'protocol': 'Blur', 'auc': round(blur_auc_mean,4)},
    {'experiment': 'Hop in-domain (same protocol)', 'protocol': 'Hop', 'auc': round(hop_baseline,4)},
    {'experiment': 'Zero-shot (Blur→Hop)', 'protocol': 'Hop', 'auc': round(hop_auc,4)},
    {'experiment': 'Fine-tuned (10% Hop labels)', 'protocol': 'Hop', 'auc': round(ft_auc,4)},
])
results.to_csv(OUT_DIR/"cross_protocol_results.csv", index=False)
print(f"\nSaved → {OUT_DIR}/cross_protocol_results.csv")
print("\n=== SUMMARY ===")
print(results.to_string(index=False))

# ── 8. Plot ─────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#3B82F6','#10B981','#EF4444','#F59E0B']
    bars = axes[0].bar(results['experiment'], results['auc'], color=colors, alpha=0.85)
    axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
    for bar, val in zip(bars, results['auc']):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_ylim([0.3, 0.9]); axes[0].set_ylabel('AUC')
    axes[0].set_title('Cross-Protocol Generalization\n(Blur NFT → Hop Bridge)', fontweight='bold')
    axes[0].set_xticklabels(results['experiment'], rotation=15, ha='right', fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # Concept diagram: feature importance comparison
    fi = pd.DataFrame({'feature': fcols,
                       'blur_gain': [full_model.feature_importance('gain')[i] for i in range(len(fcols))],
                       'hop_gain':  [ft_model.feature_importance('gain')[i] for i in range(len(fcols))]})
    x = np.arange(len(fcols)); w = 0.35
    axes[1].bar(x-w/2, fi['blur_gain']/fi['blur_gain'].sum(), w, label='Blur', color='#3B82F6', alpha=0.85)
    axes[1].bar(x+w/2, fi['hop_gain']/fi['hop_gain'].sum(), w, label='Hop', color='#10B981', alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(fcols, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('Relative Feature Importance'); axes[1].legend()
    axes[1].set_title('Feature Importance: Blur vs Hop\n(Same generic features, different weights)', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR/"cross_protocol_plot.png", dpi=150, bbox_inches='tight')
    print(f"Plot → {OUT_DIR}/cross_protocol_plot.png"); plt.close()
except Exception as e: print(f"Plot err: {e}")
print("Done!")
