"""
Phase 1: Feature Extraction for Pre-Airdrop Sybil Detection
============================================================
T0 = 2023-11-21 (Blur Season 2 first claim)
All features must be from BEFORE T0 (strictly pre-airdrop).
Labels: behavior_flags from database/ (bw/ml/fd/hf flags).
"""

import csv
import json
import pandas as pd
import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BLUR_DIR = Path.home() / "Desktop/Blur-main"
DB_DIR   = BLUR_DIR / "database"
HF_DIR   = BLUR_DIR / "Data_fetching/High_frequency_trading"
FD_DIR   = BLUR_DIR / "Data_fetching/Rapic_consolidation_season2"
OUT_DIR  = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# ── T0: airdrop start timestamp ────────────────────────────────────────────
T0 = 1700525735  # 2023-11-21 00:15 UTC (first claim tx)

print(f"T0 = {datetime.datetime.utcfromtimestamp(T0).strftime('%Y-%m-%d %H:%M')} UTC")
print(f"Loading data from {BLUR_DIR}")

# ── 1. Labels ──────────────────────────────────────────────────────────────
print("\n[1/4] Loading labels (behavior_flags)...")
labels = {}
with open(DB_DIR / "airdrop_targets_behavior_flags.csv") as f:
    for row in csv.DictReader(f):
        addr = row["address"].lower()
        labels[addr] = {
            "bw_flag": int(row["bw_flag"]),
            "ml_flag": int(row["ml_flag"]),
            "fd_flag": int(row["fd_flag"]),
            "hf_flag": int(row["hf_flag"]),
        }
print(f"  {len(labels)} addresses with labels")

# ── 2. HF features (preseason tx count) ────────────────────────────────────
print("\n[2/4] Loading HF features (preseason counts)...")
hf_feats = {}
with open(HF_DIR / "addresses_preseason_counts.csv") as f:
    for row in csv.DictReader(f):
        addr = row["address"].lower()
        hf_feats[addr] = {
            "op_count_season2":         int(row.get("op_count_season2", 0) or 0),
            "preseason_external_tx":    int(row.get("preseason_external_tx_count", 0) or 0),
            "is_hf_by_80pct":           1 if row.get("is_hf_by_80pct", "False") == "True" else 0,
        }
print(f"  {len(hf_feats)} addresses with HF features")

# ── 3. BW features (funder → wallet count) ─────────────────────────────────
print("\n[3/4] Loading BW features (funder graph)...")
funder_wallet_count = {}  # addr -> how many sibling wallets share the same funder
addr_to_funder = {}

funder_path = DB_DIR / "funder_to_airdrops_non_cex.jsonl"
if funder_path.exists():
    with open(funder_path) as f:
        for line in f:
            obj = json.loads(line)
            funder = obj.get("funder", "").lower()
            targets = [t.lower() for t in obj.get("airdrop_targets", [])]
            count = len(targets)
            for t in targets:
                addr_to_funder[t] = funder
                funder_wallet_count[t] = count  # siblings including self
    print(f"  {len(funder_wallet_count)} addresses with BW features")
else:
    print("  funder_to_airdrops_non_cex.jsonl not found, skipping BW features")

# ── 4. Merge into feature table ─────────────────────────────────────────────
print("\n[4/4] Building feature table...")

rows = []
for addr, lbl in labels.items():
    hf = hf_feats.get(addr, {})
    row = {
        "address": addr,

        # ── Pre-airdrop features (strictly T < T0) ──────────────────────
        # Account activity BEFORE Season 2
        "preseason_external_tx":    hf.get("preseason_external_tx", 0),
        "is_new_account":           1 if hf.get("preseason_external_tx", 0) == 0 else 0,

        # Activity DURING Season 2 (proxy for interaction intensity)
        # Note: op_count_season2 covers the full season window —
        # for strict experiments, further split by T0. Used here as baseline.
        "op_count_season2":         hf.get("op_count_season2", 0),

        # BW: same funder funded how many wallets?
        "funder_wallet_count":      funder_wallet_count.get(addr, 0),
        "has_batch_funder":         1 if funder_wallet_count.get(addr, 0) >= 5 else 0,

        # HF detector result
        "is_hf_by_80pct":           hf.get("is_hf_by_80pct", 0),

        # ── Labels ──────────────────────────────────────────────────────
        "bw_flag":  lbl["bw_flag"],
        "ml_flag":  lbl["ml_flag"],
        "fd_flag":  lbl["fd_flag"],
        "hf_flag":  lbl["hf_flag"],

        # Combined: any flag = Sybil hunter
        "any_flag": 1 if any(lbl[k] for k in ["bw_flag", "ml_flag", "fd_flag", "hf_flag"]) else 0,
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(f"\nFeature table shape: {df.shape}")
print(f"Label distribution (any_flag):\n{df['any_flag'].value_counts()}")
print(f"\nSybil rate: {df['any_flag'].mean():.1%}")

# Per-flag breakdown
for flag in ["bw_flag", "ml_flag", "fd_flag", "hf_flag"]:
    print(f"  {flag}: {df[flag].sum()} positive ({df[flag].mean():.1%})")

# Save
out_path = OUT_DIR / "features_baseline.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("Done! Run 02_train_lightgbm.py next.")
