# Pre-Airdrop Sybil Detection

ARTEMIS (WWW'24) detects airdrop hunters after distribution using full post-hoc graph data, achieving AUC 0.803. We ask a different question: can we detect them before the airdrop, using only what was observable on-chain before the event? The answer is yes, substantially, and months in advance.

## Core Finding

LightGBM trained on 18 behavioral features from pre-airdrop transactions achieves AUC 0.905 at T-30 (one month before Blur Season 2 distribution on 2023-11-21), and 0.895 at T-180. The AUC curve from T-0 to T-180 drops only 0.013 total, meaning the behavioral fingerprint of sybil addresses is stable long before they collect rewards. This outperforms ARTEMIS despite using strictly pre-airdrop data.

The dominant feature is NFT collection diversity. Sybil hunters spread transactions across many collections to maximize points, and this pattern is structurally different from genuine collectors regardless of how early you look. The second most time-sensitive feature is unique contract interactions: the further back you go, the more important it becomes, because it captures the early account-creation behavior of batch wallets.

## Why GNN Does Not Work Pre-Airdrop

ArtemisNet (GNN) achieves AUC 0.976 post-hoc but collapses to AUC 0.586 at T-30. The reason is architectural: ArtemisNet's graph edges include transfer events generated when the airdrop is distributed. Remove those edges and the graph is structurally incomplete. This is not a tuning issue; it is a fundamental incompatibility between graph-based post-hoc methods and the pre-airdrop setting. LightGBM on behavioral features does not have this dependency.

## Experimental Design

The 27 experiments are organized around five questions.

**Can we detect before the airdrop, and how early?** Experiments 01-03 and 12 establish the main result and temporal ablation from T-0 to T-180. Data leakage is prevented by a strict timestamp cutoff on all feature computation.

**Which features matter and why?** Experiments 04, 06, 10, and 19 characterize feature importance from three angles: LightGBM gain-based importance across time windows, group ablation (Activity / Diversity / Volume / Behavioral / DeFi), and SHAP value distributions. DeFi features contribute almost nothing (AUC 0.544 alone) because the sybil strategy centers on NFT transactions, not Blur's Blend lending product.

**How well does the model actually generalize?** Experiments 05, 08, 11, 20, 21, and 22 test generalization from multiple directions: precision-recall thresholds for deployment, temporal and population splits, sybil subtype clustering (three types: retail hunter, mid-volume, and six hyperactive bots), leave-one-flag-out generalization by sybil strategy, and probability calibration. The critical failure case is flag-type generalization: when BW (high-value buyer) sybils are excluded from training, AUC drops to 0.047, because the model never learned to associate high buy_value with sybil behavior.

**Does the method transfer to other protocols?** Experiments 14-18 and 24 test transfer across Blur, Hop Protocol, Gitcoin GR15, and LayerZero. Zero-shot transfer with protocol-specific features yields AUC 0.22-0.47, which initially looks like failure. Experiment 17 shows this is primarily feature mismatch: with a five-feature common set (tx_count, total_volume, wallet_age_days, unique_contracts, active_span_days), Blur-to-Hop zero-shot improves from 0.38 to 0.78. Experiment 24 adds LayerZero as a fourth protocol: same-class bridge transfer (Hop to LZ) achieves 0.567, better than cross-domain transfer (Blur to LZ at 0.434), supporting the hypothesis that protocol category is a meaningful predictor of transfer difficulty.

**Are the weaknesses addressable?** Experiments 25-27 directly address the three main limitations. Isolation Forest (unsupervised) raises detection of the unseen BW type from 0.047 to 0.916, because BW sybils are statistical outliers in feature space regardless of labeling. A two-stage deployment (supervised LightGBM plus unsupervised IF) is robust to novel sybil strategies. Label noise experiments show AUC drops only 0.014 under 20% label corruption, meaning the model is deployable even when the official blacklist is imperfect. Rule-based baselines (wallet age, transaction count thresholds) peak at AUC 0.610, far below LightGBM's 0.905, confirming that simple heuristics cannot capture the composite behavioral patterns that distinguish sybils from genuine heavy users.

## Dataset

Primary: Blur Season 2 on-chain transactions, Feb-Nov 2023, 3.2M transactions, 251K addresses.
Cross-protocol: Hop Protocol (204K addresses, 14K sybil), Gitcoin GR15 (39.9K addresses), LayerZero ZRO (29.8K addresses fetched, 19.5K active after filtering).
Labels source: official protocol sybil lists (Blur airdrop2_targets, Hop Sybil Hunter program, LayerZero Proof of Sybil campaign).

## Experiments

| Script | Experiment |
|--------|------------|
| 01_build_features.py | Feature extraction at each temporal cutoff (18 behavioral features) |
| 02_train_lightgbm.py | LightGBM baseline, 5-fold CV, comparison with ARTEMIS |
| 03_temporal_ablation.py | AUC at T-0, T-7, T-14, T-30, T-60, T-90 |
| 04_feature_importance.py | Feature importance across time windows |
| 05_pr_analysis.py | Precision-recall curve and optimal deployment threshold |
| 06_ablation_features.py | Feature group ablation: Activity / Diversity / Volume / Behavioral / DeFi |
| 07_graph_features.py | Graph-augmented features: behavioral (0.904) vs behavioral+graph (0.905) |
| 08_generalization.py | Temporal split (T-90 train, T-30 test) and population split (50% unseen sybil) |
| 09_artemis_gnn.py | ArtemisNet GNN post-hoc reproduction (AUC 0.976) |
| 10_shap_analysis.py | SHAP beeswarm attribution |
| 11_sybil_clustering.py | K-means clustering of sybil subtypes (K=3) |
| 12_extended_temporal.py | Extended ablation to T-120, T-150, T-180 |
| 13_adversarial_robustness.py | Evasion robustness under diversity reduction |
| 14_cross_protocol.py | Blur-to-Hop zero-shot and fine-tuned transfer |
| 15_gnn_preairdrop.py | ArtemisNet at T-30 (AUC 0.586, architectural failure) |
| 16_lopo_crossprotocol.py | Three-protocol LOPO with protocol-specific features (AUC 0.22-0.47) |
| 16b_lopo_blur_hop.py | Blur-Hop label-budget study: 1% labels recovers AUC 0.982 |
| 17_common_features_lopo.py | LOPO with five-protocol-agnostic features (Blur-Hop: 0.38 to 0.78) |
| 18_gitcoin_feature_importance.py | Gitcoin domain analysis: gitcoin_donations importance = 0% |
| 19_shap_temporal.py | SHAP comparison T-90 vs T-7: wallet_age vs recent_activity shift |
| 20_clustering_k_selection.py | K selection: silhouette optimal K=2, semantic choice K=3 |
| 21_flag_type_generalization.py | Leave-one-flag-out: BW=0.047, HF=0.550, FD=0.110, ML=0.241 |
| 22_calibration.py | Isotonic calibration: Brier score 0.120 to 0.091, AUC unchanged |
| 23_adversarial_cost.py | Economic cost of evasion: 90% diversity cut costs 67% of points, AUC -0.6% |
| 24_layerzero_lopo.py | LayerZero as fourth protocol: Hop-to-LZ 0.567, Blur-to-LZ 0.434 |
| 25_openworld_detection.py | Isolation Forest rescues BW detection: 0.047 to 0.916 |
| 26_label_noise.py | Label noise robustness: 20% corruption drops AUC only 0.014 |
| 27_rule_baselines.py | Rule baselines peak at 0.610; LightGBM 0.905; RF 0.897 |

Experiment map with figures: [MINDMAP.html](https://tyche1107.github.io/pre-airdrop-detection/MINDMAP.html)

## Reproducibility

All temporal cutoffs are enforced by filtering on raw transaction timestamps before any feature computation. Features at T-k use only transactions with timestamp less than (T0 minus k days in seconds). Labels are applied after feature computation and are never used to select which transactions to include. The Blur T0 is 1700525735 (first Season 2 claim transaction, 2023-11-21 00:15:35 UTC).
