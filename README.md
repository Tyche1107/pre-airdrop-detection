# Pre-Airdrop Sybil Detection

Detecting airdrop hunters before an airdrop using on-chain behavioral features. Built on Blur NFT Season 2 data with validation on LayerZero.

## Key Results

| Method | Data timing | Dataset | AUC |
|--------|-------------|---------|-----|
| ARTEMIS GNN (WWW'24) | Post-airdrop | Blur | 0.803 |
| LightGBM T-30 | Pre-airdrop 30d | Blur (53K) | **0.793** |
| LightGBM T-90 | Pre-airdrop 90d | Blur (53K) | **0.789** |
| ArtemisNet GNN T-30 | Pre-airdrop 30d | Blur (53K) | 0.586 |
| LightGBM T-30 | Pre-airdrop 30d | LayerZero | **0.946** |

Signal stability: AUC drops 0.009 (T-0 to T-90) on Blur, 0.0006 on LayerZero.

## Datasets

**Blur NFT Season 2** (main dataset)
- `TXS2_1662_1861.csv` — 3.2M NFT transactions, all Blur users, Feb-Nov 2023
- `addresses_all_with_loaylty_blend_blur_label319.csv` — Blend/LP extended features, 251K addresses
- `airdrop_targets_behavior_flags.csv` — ground truth, 53,482 Season 2 recipients with 4 behavior flags:
  - `bw_flag`: linked to non-CEX funder with >= 5 targets
  - `ml_flag`: appears in NFT loop cycle of length >= 5
  - `fd_flag`: rapid claim-to-transfer consolidation
  - `hf_flag`: high-frequency trading (80th percentile rule)
  - Sybil (any flag = 1): **9,817** | Normal (all flags = 0): 43,665
  - Training population: 53,482 airdrop recipients only (not all TXS2 users)

**LayerZero**
- `layerzero/lz_temporal_features.csv` — 29,870 ETH-chain addresses (Alchemy API), T-30/60/90 cutoffs
- Labels: LayerZero official Proof-of-Sybil list
- Note: ARB and POLY chains pending Alchemy activation

**Cross-protocol**
- Hop: `hop/metadata.csv`, `hop/sybil_addresses.csv` (official sybil list)
- Gitcoin: `gitcoin/onchain_features.csv` — 39,962 addresses with SADScore labels

## Experiments

### Q1: Can we detect before the airdrop?

| Exp | Description | Data | Model |
|-----|-------------|------|-------|
| 03 | Temporal ablation T-0 to T-90 | TXS2, 53K recipients | LightGBM 5-fold |
| 12 | Extended ablation T-120 to T-180 | TXS2, 53K recipients | LightGBM |
| 05 | Precision-recall curve | TXS2, 53K | LightGBM |
| 09 | ARTEMIS GNN reproduction | Artemis/*.pt | GraphSAGE |
| 15 | GNN pre-airdrop comparison | TXS2, 53K | GNN |
| 28 | LayerZero temporal ablation | lz_temporal_features.csv | LightGBM |

### Q2: Which features work?

| Exp | Description | Data | Model |
|-----|-------------|------|-------|
| 01 | Feature construction | TXS2 + ext | Outputs nft_feats_labeled_T*.csv |
| 04 | Feature importance across windows | TXS2, 53K | LightGBM SHAP |
| 06 | Feature group ablation | TXS2, 53K | LightGBM |
| 07 | Graph feature augmentation | TXS2 + fund_graph_edges.csv | LightGBM |
| 10 | SHAP beeswarm | TXS2, 53K | LightGBM |
| 19 | SHAP temporal comparison | nft_feats_labeled_T30/T90.csv | LightGBM |

### Q3: How robust?

| Exp | Description | Data | Model |
|-----|-------------|------|-------|
| 08 | Time + population generalization | TXS2, 53K | LightGBM |
| 26 | Label noise robustness | nft_feats_labeled_T30.csv | LightGBM |
| 13/23 | Adversarial robustness + evasion cost | nft_feats_labeled_T30.csv | LightGBM |
| 22 | Probability calibration | nft_feats_labeled_T30.csv | LightGBM + Isotonic |

### Q4: Does it transfer across protocols?

| Exp | Description | Protocols |
|-----|-------------|-----------|
| 28 | LayerZero temporal ablation | LayerZero |
| 24 | LayerZero LOPO | Blur, Hop, LZ |
| 17 | Common-feature LOPO | Blur, Gitcoin, Hop |
| 16 | Protocol-specific LOPO | Blur, Gitcoin, Hop |
| 14/16b | Blur to Hop transfer | Blur, Hop |
| 18 | Gitcoin in-domain analysis | Gitcoin |

### Supporting: Baselines / Sybil Analysis / Calibration

| Exp | Description | Data | Model |
|-----|-------------|------|-------|
| 27 | Rule baselines | nft_feats_labeled_T30.csv | Rules / LR / RF / LightGBM |
| 11 | Sybil strategy clustering | TXS2, sybil-only (9,817) | K-means K=3 |
| 20 | K selection | nft_feats_T30, is_sybil=1 | Elbow + Silhouette |
| 21 | Per-flag-type leave-one-out | nft_feats_labeled_T30.csv | LightGBM |
| 25 | Open-world detection (BW holdout) | nft_feats_labeled_T30.csv | LightGBM + Isolation Forest |

## Reproducibility

```bash
cd pre-airdrop-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build labeled feature files (requires raw data)
python 00_build_nft_feats_labeled.py

# Main temporal ablation
python 03_temporal_ablation.py

# All other experiments
for i in 04 05 06 07 08 09 10 11 12 13 15 19 20 21 22 23 25 26 27 28; do
    python ${i}_*.py
done
```

Raw data paths default to `~/Desktop/dataset/`. Adjust `DATA_DIR` at the top of each script.
