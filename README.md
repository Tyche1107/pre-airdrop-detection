# Pre-Airdrop Sybil Detection

Can on-chain behavior before an airdrop identify sybil hunters? This project builds and evaluates a pre-airdrop sybil detection system using behavioral features extracted from Blur NFT Season 2 transaction data.

## Key Results

| Method | Data | AUC |
|--------|------|-----|
| ARTEMIS GNN (WWW'24 baseline) | Post-airdrop | 0.803 |
| LightGBM, T-30 (ours) | Pre-airdrop, 30 days before | **0.793** |
| LightGBM, T-90 (ours) | Pre-airdrop, 90 days before | **0.789** |
| ArtemisNet GNN, T-30 (ours) | Pre-airdrop, 30 days before | 0.586 |
| LightGBM, LayerZero T-30 | Pre-airdrop, second protocol | **0.946** |

Signal stability: T-0 to T-90 AUC drops only 0.009 on Blur, 0.0006 on LayerZero.

## Datasets

**Blur NFT Season 2**
- Transactions: `TXS2_1662_1861.csv` — 3.2M NFT trades, Feb-Nov 2023
- Extended features: `addresses_all_with_loaylty_blend_blur_label319.csv` — Blend/LP features
- Ground truth: `airdrop_targets_behavior_flags.csv` — 53,482 airdrop recipients with behavior flags
  - `bw_flag`: linked to non-CEX funder with >= 5 targets
  - `ml_flag`: appears in NFT loop cycle of length >= 5
  - `fd_flag`: rapid claim-to-transfer consolidation
  - `hf_flag`: high-frequency trading by 80th percentile rule
  - Sybil (any flag = 1): 9,817 | Normal (all flags = 0): 43,665

**LayerZero**
- `layerzero/lz_temporal_features.csv` — 29,870 addresses, ETH chain (Alchemy API)
- Labels: LayerZero official Proof-of-Sybil list

**Cross-protocol**
- Hop: `hop/metadata.csv`, `hop/sybil_addresses.csv`
- Gitcoin: `gitcoin/onchain_features.csv` (39,962 addresses, SADScore labels)

## Experiments

### Main Results
| Exp | Description | Data | Model |
|-----|-------------|------|-------|
| 03 | Temporal ablation T-0 to T-90 | TXS2, 53K recipients | LightGBM 5-fold |
| 12 | Extended ablation T-120 to T-180 | TXS2, 53K recipients | LightGBM |
| 05 | Precision-recall analysis | TXS2, 53K recipients | LightGBM |
| 28 | LayerZero temporal ablation | lz_temporal_features.csv | LightGBM |

### Feature Analysis
| Exp | Description | Data | Model |
|-----|-------------|------|-------|
| 01 | Feature construction | TXS2 + ext | Output: nft_feats_labeled_T*.csv |
| 04 | Feature importance across windows | TXS2, 53K | LightGBM SHAP |
| 06 | Feature group ablation | TXS2, 53K | LightGBM |
| 10 | SHAP beeswarm | TXS2, 53K | LightGBM |
| 19 | SHAP temporal comparison | nft_feats_labeled_T30/90.csv | LightGBM |
| 07 | Graph feature augmentation | TXS2 + fund_graph_edges.csv | LightGBM |

### Robustness
| Exp | Description | Data | Model |
|-----|-------------|------|-------|
| 08 | Time and population generalization | TXS2, 53K | LightGBM |
| 26 | Label noise robustness | nft_feats_labeled_T30.csv | LightGBM |
| 13/23 | Adversarial robustness and evasion cost | nft_feats_labeled_T30.csv | LightGBM |
| 22 | Probability calibration | nft_feats_labeled_T30.csv | LightGBM + Isotonic |

### Sybil Characterization
| Exp | Description | Data | Model |
|-----|-------------|------|-------|
| 11 | Sybil strategy clustering | TXS2, sybil-only (9,817) | K-means K=3 |
| 20 | K selection | nft_feats_T30, is_sybil=1 | Elbow + Silhouette |
| 21 | Per-flag-type leave-one-out | nft_feats_labeled_T30.csv | LightGBM |
| 25 | Open-world detection (BW holdout) | nft_feats_labeled_T30.csv | LightGBM + IF |

### Baselines
| Exp | Description | Data | Model |
|-----|-------------|------|-------|
| 09 | ARTEMIS GNN reproduction | Artemis/*.pt | GraphSAGE |
| 15 | GNN pre-airdrop comparison | TXS2, 53K | GNN |
| 27 | Rule baselines | nft_feats_labeled_T30.csv | Rules / LR / RF / LightGBM |

### Cross-Protocol
| Exp | Description | Protocols |
|-----|-------------|-----------|
| 14/16b | Blur to Hop transfer | Blur, Hop |
| 16 | 3-protocol LOPO (protocol-specific features) | Blur, Gitcoin, Hop |
| 17 | 3-protocol LOPO (common features) | Blur, Gitcoin, Hop |
| 18 | Gitcoin in-domain analysis | Gitcoin |
| 24 | LayerZero LOPO | Blur, Hop, LayerZero |

## Reproducibility

```bash
cd pre-airdrop-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build feature files (requires TXS2 and behavior_flags)
python 00_build_nft_feats_labeled.py

# Main temporal ablation
python 03_temporal_ablation.py

# Run all experiments
for i in 04 05 06 07 08 09 10 11 12 13 15 19 20 21 22 23 25 26 27; do
    python ${i}_*.py
done
```

Data paths default to `~/Desktop/dataset/`. Adjust `DATA_DIR` at the top of each script if needed.

## Notes

- Training population is restricted to the 53,482 Season 2 airdrop recipients, not all TXS2 users.
- LayerZero features are ETH-chain only; ARB and POLY pending Alchemy activation.
- ARTEMIS GNN requires post-airdrop transfer edges; pre-airdrop graph is incomplete, causing AUC to drop to 0.586 at T-30.
