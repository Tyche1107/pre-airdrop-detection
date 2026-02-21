"""
09_artemis_gnn.py — Artemis GNN vs LightGBM Comparison
"""
import torch, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
sys.path.insert(0, str(Path.home() / 'Desktop/Blur-main/Artemis'))
from model.artemis import ArtemisNet

DATA_DIR = Path('/tmp/artemis_data')
OUT_DIR  = Path.home() / 'Desktop/pre-airdrop-detection/data'

print("Loading tensors (CPU)...")
nb  = torch.load(DATA_DIR/'node_basic_features.pt',             map_location='cpu', weights_only=True)
na  = torch.load(DATA_DIR/'node_advanced_features.pt',          map_location='cpu', weights_only=True)
ei  = torch.load(DATA_DIR/'edge_index.pt',                      map_location='cpu', weights_only=True).long()
ef  = torch.load(DATA_DIR/'base_edge_features.pt',              map_location='cpu', weights_only=True)
ne  = torch.load(DATA_DIR/'nft_multimodal_bmbedding_features.pt', map_location='cpu', weights_only=True)
y   = torch.load(DATA_DIR/'y.pt',                               map_location='cpu', weights_only=True).float()

X   = torch.cat([nb, na], dim=1).float()
EA  = torch.cat([ef, ne], dim=1).float()
N   = X.shape[0]
y_np = y.numpy()
print(f"  {N:,} nodes | {int(y.sum()):,} Sybil ({y.mean()*100:.1f}%) | edges {ei.shape[1]:,}")
print(f"  Node feat: {X.shape[1]}  Edge feat: {EA.shape[1]}")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs, fold_f1s = [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(np.zeros(N), y_np)):
    print(f"\nFold {fold+1}/5 ...")
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    train_mask[tr_idx] = True
    val_mask[val_idx]  = True

    model = ArtemisNet(X.shape[1], EA.shape[1], 32)
    pos_w  = torch.tensor([(train_mask & (y==0)).sum().float() / max((train_mask & (y==1)).sum().float(), 1)])
    crit   = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt    = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

    best_auc, best_state, patience_cnt = 0, None, 0
    ei3 = (ei, ei, ei)
    ea3 = (EA, EA, EA)

    for epoch in range(30):
        model.train()
        opt.zero_grad()
        out  = model(X, ei3, ea3)
        loss = crit(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 10 == 0 or epoch == 99:
            model.eval()
            with torch.no_grad():
                proba = torch.sigmoid(model(X, ei3, ea3)[val_mask]).numpy()
            auc = roc_auc_score(y_np[val_idx], proba)
            print(f"  Epoch {epoch:3d}  loss={loss.item():.4f}  val_AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_state = {k:v.clone() for k,v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= 3:
                print("  Early stop")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(X, ei3, ea3)[val_mask]).numpy()
    auc = roc_auc_score(y_np[val_idx], proba)
    preds = (proba > 0.5).astype(int)
    f1  = f1_score(y_np[val_idx], preds, zero_division=0)
    fold_aucs.append(auc); fold_f1s.append(f1)
    print(f"  Fold {fold+1} best: AUC={auc:.4f}  F1={f1:.4f}")

mean_auc = np.mean(fold_aucs)
mean_f1  = np.mean(fold_f1s)
print(f"\n{'='*55}")
print(f"ArtemisNet 5-fold CV:  AUC={mean_auc:.4f}  F1={mean_f1:.4f}")
print(f"\nComparison:")
print(f"  ARTEMIS paper (post-hoc):         0.803")
print(f"  Our LightGBM T-30 (pre-airdrop):  0.9045")
print(f"  ArtemisNet this run:              {mean_auc:.4f}")

pd.DataFrame([
    {'model':'ARTEMIS (paper)',      'auc':0.803,     'note':'post-hoc GNN, WWW24'},
    {'model':'LightGBM T-30',        'auc':0.9045,    'note':'pre-airdrop behavioral'},
    {'model':'ArtemisNet (our run)', 'auc':mean_auc,  'note':'GNN, same data'},
]).to_csv(OUT_DIR/'gnn_comparison.csv', index=False)
print(f"Saved → {OUT_DIR}/gnn_comparison.csv")
print("Done!")
