"""
Temporal Ablation Study
========================
Core experiment: at T-N days before airdrop, what's detection performance?
Requires per-transaction timestamps from Google Drive NFT data.

Input: nft_transactions with columns [address, timestamp, ...]
T0 = 1700525735 (2023-11-21 00:15 UTC, first Blur S2 claim)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from pathlib import Path
import json, csv

DATA_DIR  = Path(__file__).parent / "data"
FIG_DIR   = Path(__file__).parent / "figures"
BLUR_DIR  = Path.home() / "Desktop/Blur-main"
FIG_DIR.mkdir(exist_ok=True)

T0 = 1700525735  # 2023-11-21 00:15 UTC

# ── Load labels ──────────────────────────────────────────────────────────
labels = {}
with open(BLUR_DIR / "database/airdrop_targets_behavior_flags.csv") as f:
    for row in csv.DictReader(f):
        addr = row["address"].lower()
        labels[addr] = 1 if any(int(row[k]) for k in ["bw_flag","ml_flag","fd_flag","hf_flag"]) else 0

# ── Load base features (always available regardless of window) ────────────
hf_feats = {}
with open(BLUR_DIR / "Data_fetching/High_frequency_trading/addresses_preseason_counts.csv") as f:
    for row in csv.DictReader(f):
        addr = row["address"].lower()
        hf_feats[addr] = {
            "preseason_external_tx": int(row.get("preseason_external_tx_count", 0) or 0),
        }

funder_count = {}
funder_path = BLUR_DIR / "database/funder_to_airdrops_non_cex.jsonl"
if funder_path.exists():
    with open(funder_path) as f:
        for line in f:
            obj = json.loads(line)
            targets = [t.lower() for t in obj.get("airdrop_targets", [])]
            for t in targets:
                funder_count[t] = len(targets)

# ── Load NFT transaction data (Google Drive) ──────────────────────────────
# Expected: CSV with columns [address, timestamp, nft_buy, nft_sell, ...]
# OR JSONL with per-transaction records
# TODO: Update path when data is downloaded

NFT_TX_PATH = DATA_DIR / "nft_transactions.csv"  # update when available

def load_nft_features_for_window(cutoff_ts):
    """
    Extract NFT-based features using only transactions BEFORE cutoff_ts.
    Returns dict: address -> {nft_tx_count, buy_count, sell_count, avg_hold_sec, ...}
    """
    feats = {}
    if not NFT_TX_PATH.exists():
        print(f"  [!] NFT data not found at {NFT_TX_PATH}, skipping NFT features")
        return feats

    df_nft = pd.read_csv(NFT_TX_PATH)
    # Filter to pre-cutoff transactions
    df_pre = df_nft[df_nft["timestamp"] < cutoff_ts]

    for addr, grp in df_pre.groupby("address"):
        feats[addr.lower()] = {
            "nft_tx_count":   len(grp),
            "nft_buy_count":  (grp.get("type", pd.Series()) == "buy").sum() if "type" in grp else 0,
            "nft_sell_count": (grp.get("type", pd.Series()) == "sell").sum() if "type" in grp else 0,
            "nft_active_days": grp["timestamp"].nunique() if "timestamp" in grp else 0,
        }
    return feats

def build_feature_df(cutoff_ts):
    """Build feature table using data available T days before T0."""
    nft_feats = load_nft_features_for_window(cutoff_ts)
    rows = []
    for addr, label in labels.items():
        hf = hf_feats.get(addr, {})
        nft = nft_feats.get(addr, {})
        rows.append({
            "address": addr,
            # Base features (always available)
            "preseason_external_tx": hf.get("preseason_external_tx", 0),
            "is_new_account":        1 if hf.get("preseason_external_tx", 0) == 0 else 0,
            "funder_wallet_count":   funder_count.get(addr, 0),
            "has_batch_funder":      1 if funder_count.get(addr, 0) >= 5 else 0,
            # NFT features (time-limited to cutoff_ts)
            "nft_tx_count":    nft.get("nft_tx_count", 0),
            "nft_buy_count":   nft.get("nft_buy_count", 0),
            "nft_sell_count":  nft.get("nft_sell_count", 0),
            "nft_active_days": nft.get("nft_active_days", 0),
            # Label
            "label": label,
        })
    return pd.DataFrame(rows)

def evaluate_window(days_before):
    cutoff = T0 - days_before * 86400
    print(f"\nWindow: T-{days_before} days (cutoff before {pd.Timestamp(cutoff, unit='s').strftime('%Y-%m-%d')})")

    df = build_feature_df(cutoff)
    feature_cols = [c for c in df.columns if c not in ["address", "label"]]
    X = df[feature_cols].fillna(0).values
    y = df["label"].values

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1s, aucs, precs, recs = [], [], [], []
    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, scale_pos_weight=pos_weight,
            random_state=42, verbose=-1
        )
        model.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

        prob = model.predict_proba(X[val_idx])[:, 1]
        # Find best threshold
        from sklearn.metrics import precision_recall_curve
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y[val_idx], prob)
        f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
        best = np.argmax(f1_arr)
        thr = thr_arr[best] if best < len(thr_arr) else 0.5
        pred = (prob >= thr).astype(int)

        f1s.append(f1_score(y[val_idx], pred, zero_division=0))
        aucs.append(roc_auc_score(y[val_idx], prob))
        precs.append(precision_score(y[val_idx], pred, zero_division=0))
        recs.append(recall_score(y[val_idx], pred, zero_division=0))

    result = {
        "days": days_before,
        "f1":   np.mean(f1s),   "f1_std":   np.std(f1s),
        "auc":  np.mean(aucs),  "auc_std":  np.std(aucs),
        "precision": np.mean(precs),
        "recall":    np.mean(recs),
    }
    print(f"  F1={result['f1']:.3f}±{result['f1_std']:.3f}  "
          f"AUC={result['auc']:.3f}  "
          f"P={result['precision']:.3f}  R={result['recall']:.3f}")
    return result

# ── Run ablation ──────────────────────────────────────────────────────────
# Coarse scan first
WINDOWS_COARSE = [7, 14, 30, 60, 90]

print("=" * 55)
print("Temporal Ablation Study")
print(f"T0 = 2023-11-21, scanning {WINDOWS_COARSE} days before")
print("=" * 55)

results = [evaluate_window(d) for d in WINDOWS_COARSE]
df_res = pd.DataFrame(results)
df_res.to_csv(DATA_DIR / "temporal_ablation_results.csv", index=False)

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.errorbar(df_res["days"], df_res["f1"], yerr=df_res["f1_std"],
            fmt='bo-', linewidth=2, markersize=8, capsize=4)
ax.set_xlabel("Days Before Airdrop (T)", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("Detection Performance vs. Lead Time", fontsize=13, fontweight='bold')
ax.invert_xaxis()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

ax = axes[1]
ax.plot(df_res["days"], df_res["precision"], 'r^--', label='Precision', linewidth=2)
ax.plot(df_res["days"], df_res["recall"],    'gs--', label='Recall',    linewidth=2)
ax.plot(df_res["days"], df_res["auc"],       'bo-',  label='AUC',       linewidth=2)
ax.set_xlabel("Days Before Airdrop (T)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Precision / Recall / AUC vs. Lead Time", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(FIG_DIR / "temporal_ablation.png", dpi=150, bbox_inches='tight')
print(f"\nFigure saved: figures/temporal_ablation.png")
print("\n✅ Temporal ablation complete!")
print("\nKey question: at which T does F1 drop significantly?")
print("That T is the 'earliest detection horizon' — the paper's core finding.")
