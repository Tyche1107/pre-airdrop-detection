"""15_gnn_preairdrop.py — ArtemisNet on PRE-AIRDROP data only (T-30)
Fair comparison: same data window as LightGBM T-30.
Build address transaction graph from TXS2 restricted to T-30.
"""
import sys, torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/adelinewen/Desktop/dataset/blurtx/dataset/blurtx/dataset")
OUT_DIR  = Path("/Users/adelinewen/Desktop/pre-airdrop-detection/data")
T0       = 1700525735
CUTOFF   = T0 - 30 * 86400  # T-30

print("=== GNN Pre-Airdrop Comparison (T-30) ===")
print(f"Data cutoff: T-30 = {pd.Timestamp(CUTOFF, unit='s').date()}")
sys.stdout.flush()

# ── 1. Load & filter TXS2 to T-30 ───────────────────────────────────────────
print("\n[1] Loading TXS2 (T-30 only)...")
txs = pd.read_csv(DATA_DIR/"TXS2_1662_1861.csv",
    usecols=['send','receive','timestamp','trade_price','contract_address'],
    dtype={'timestamp':'Int64','trade_price':'float32'}, low_memory=False)
txs['ts'] = txs['timestamp'] // 1000
txs = txs[txs['ts'] < CUTOFF].copy()
for c in ['send','receive']: txs[c] = txs[c].str.lower().fillna('')
print(f"  {len(txs):,} transactions (pre-T30)")
sys.stdout.flush()

# ── 2. Build node features (same as LightGBM) ───────────────────────────────
print("[2] Building node features...")
ext = pd.read_csv(DATA_DIR/"addresses_all_with_loaylty_blend_blur_label319.csv")
ext['address'] = ext['address'].str.lower()

buys  = txs[['send','contract_address','trade_price','ts']].rename(columns={'send':'addr'})
sells = txs[['receive','contract_address','trade_price','ts']].rename(columns={'receive':'addr'})
all_a = pd.concat([buys[['addr']], sells[['addr']]]).drop_duplicates()
all_a = all_a[all_a['addr'].str.startswith('0x')].copy()

buy_s  = buys.groupby('addr').agg(buy_count=('addr','count'), buy_value=('trade_price','sum'),
          buy_collections=('contract_address','nunique'), buy_last=('ts','max'), buy_first=('ts','min')).reset_index()
sell_s = sells.groupby('addr').agg(sell_count=('addr','count'), sell_value=('trade_price','sum')).reset_index()

feat = all_a.merge(buy_s, on='addr', how='left').merge(sell_s, on='addr', how='left').fillna(0)
feat['total_trades']  = feat['buy_count'] + feat['sell_count']
feat['sell_ratio']    = feat['sell_count'] / (feat['total_trades'] + 1e-6)
feat['wallet_age']    = (CUTOFF - feat['buy_first'].clip(lower=0)) / 86400
feat['pnl_proxy']     = feat['sell_value'] - feat['buy_value']
feat['avg_vol']       = (feat['buy_value'] + feat['sell_value']) / (feat['total_trades'] + 1e-6)
feat = feat[feat['total_trades'] > 0].merge(
    ext[['address','unique_interactions','blend_in_count','blend_out_count','ratio']],
    left_on='addr', right_on='address', how='left').fillna(0)

_flags = pd.read_csv(DATA_DIR / "airdrop_targets_behavior_flags.csv")
targets = set(_flags[(_flags['bw_flag']==1)|(_flags['ml_flag']==1)|(_flags['fd_flag']==1)|(_flags['hf_flag']==1)]['address'].str.lower())
del _flags
feat['label'] = feat['addr'].isin(targets).astype(int)

FEAT_COLS = ['buy_count','sell_count','total_trades','buy_value','sell_value','pnl_proxy',
             'buy_collections','sell_ratio','wallet_age','avg_vol',
             'unique_interactions','blend_in_count','blend_out_count','ratio']
fcols = [c for c in FEAT_COLS if c in feat.columns]
feat = feat.reset_index(drop=True)
for c in fcols: feat[c] = pd.to_numeric(feat[c], errors='coerce').fillna(0)
addr2idx = {a: i for i, a in enumerate(feat['addr'])}
print(f"  {len(feat):,} nodes | {feat['label'].sum():,} Sybil ({feat['label'].mean()*100:.1f}%)")
sys.stdout.flush()

# ── 3. Build edge list (buyer → seller per transaction) ─────────────────────
print("[3] Building transaction graph...")
# Filter to addresses in our node set
txs_filt = txs[(txs['send'].isin(addr2idx)) & (txs['receive'].isin(addr2idx))].copy()
txs_filt = txs_filt[txs_filt['send'] != txs_filt['receive']]  # remove self-loops

src = txs_filt['send'].map(addr2idx).values
dst = txs_filt['receive'].map(addr2idx).values
edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

# Edge features: normalized price, recency
prices = torch.tensor(txs_filt['trade_price'].values, dtype=torch.float).unsqueeze(1)
recency = torch.tensor((CUTOFF - txs_filt['ts'].values) / 86400, dtype=torch.float).unsqueeze(1)
edge_attr = torch.cat([prices, recency], dim=1)

X = torch.tensor(feat[fcols].values.astype(np.float32), dtype=torch.float)
y = torch.tensor(feat['label'].values, dtype=torch.long)
print(f"  {edge_index.shape[1]:,} edges | {X.shape[1]} node features")
sys.stdout.flush()

# ── 4. GNN Model (same architecture as 09_artemis_gnn.py) ───────────────────
class ArtemisNet(torch.nn.Module):
    def __init__(self, in_ch, hidden=64, heads=4, drop=0.3):
        super().__init__()
        self.gat1 = GATConv(in_ch, hidden, heads=heads, dropout=drop, concat=True)
        self.gat2 = GATConv(hidden*heads, hidden, heads=1, dropout=drop, concat=False)
        self.mlp  = torch.nn.Sequential(
            torch.nn.Linear(hidden, 32), torch.nn.ReLU(),
            torch.nn.Dropout(drop), torch.nn.Linear(32, 2))
    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return self.mlp(x)

# ── 5. 5-fold CV ────────────────────────────────────────────────────────────
print("[4] Training ArtemisNet (5-fold CV, T-30 graph)...")
kf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_np = y.numpy()
fold_aucs = []
data = Data(x=X, edge_index=edge_index, y=y)
spw  = (y_np==0).sum() / max((y_np==1).sum(), 1)
class_weights = torch.tensor([1.0, spw], dtype=torch.float)

for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(len(y_np)), y_np)):
    tr_mask  = torch.zeros(len(y_np), dtype=torch.bool); tr_mask[tr_idx]  = True
    val_mask = torch.zeros(len(y_np), dtype=torch.bool); val_mask[val_idx] = True

    model = ArtemisNet(X.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

    best_auc, patience, wait = 0.0, 30, 0
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[tr_mask], data.y[tr_mask], weight=class_weights)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            prob = F.softmax(out, dim=1)[:,1].numpy()
        auc = roc_auc_score(y_np[val_idx], prob[val_idx])
        if auc > best_auc:
            best_auc = auc; wait = 0
        else:
            wait += 1
            if wait >= patience: break

        if epoch % 20 == 0:
            print(f"  Fold {fold+1}  Epoch {epoch:3d}  loss={loss.item():.4f}  val_AUC={auc:.4f}")
            sys.stdout.flush()

    fold_aucs.append(best_auc)
    print(f"  Fold {fold+1} best: AUC={best_auc:.4f}")
    sys.stdout.flush()

mean_auc = float(np.mean(fold_aucs))
std_auc  = float(np.std(fold_aucs))
print(f"\n{'='*50}")
print(f"ArtemisNet PRE-AIRDROP (T-30): AUC={mean_auc:.4f} ± {std_auc:.4f}")
print(f"{'='*50}")

# ── 6. Update comparison table ───────────────────────────────────────────────
df = pd.DataFrame([
    {'model': 'ARTEMIS (paper)',             'auc': 0.803,      'data': 'post-hoc', 'note': 'WWW24 reported'},
    {'model': 'LightGBM (T-30, ours)',       'auc': 0.9045,     'data': 'pre-airdrop T-30', 'note': 'behavioral features'},
    {'model': 'LightGBM (T-90, ours)',       'auc': 0.9014,     'data': 'pre-airdrop T-90', 'note': 'behavioral features'},
    {'model': 'ArtemisNet GNN (post-hoc)',   'auc': 0.9756,     'data': 'post-hoc', 'note': 'full graph, 5-fold CV'},
    {'model': 'ArtemisNet GNN (T-30, ours)','auc': round(mean_auc,4), 'data': 'pre-airdrop T-30', 'note': 'fair comparison'},
])
df.to_csv(OUT_DIR/"model_comparison_full.csv", index=False)
print(f"\nFull comparison saved → {OUT_DIR}/model_comparison_full.csv")
print(df[['model','data','auc']].to_string(index=False))

# ── 7. Plot ──────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt, matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {'post-hoc':'#94A3B8', 'pre-airdrop T-30':'#3B82F6', 'pre-airdrop T-90':'#6366F1'}
    bars = ax.barh(df['model'], df['auc'],
                   color=[colors.get(d,'#94A3B8') for d in df['data']], alpha=0.85)
    ax.axvline(0.803, color='red', linestyle='--', alpha=0.6, label='ARTEMIS baseline (0.803)')
    for bar, val in zip(bars, df['auc']):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    ax.set_xlim([0.5, 1.02]); ax.set_xlabel('AUC (5-fold CV)', fontsize=12)
    ax.set_title('Pre-Airdrop vs Post-Hoc Sybil Detection\n(Fair comparison on same data windows)',
                 fontsize=13, fontweight='bold')
    patches = [mpatches.Patch(color=v, label=k) for k,v in colors.items() if k in df['data'].values]
    ax.legend(handles=patches, fontsize=10); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR/"model_comparison_plot.png", dpi=150, bbox_inches='tight')
    print(f"Plot → {OUT_DIR}/model_comparison_plot.png"); plt.close()
except Exception as e: print(f"Plot err: {e}")
print("Done!")
