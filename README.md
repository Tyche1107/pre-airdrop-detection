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

Experiment map with figures and results: [MINDMAP.html](https://tyche1107.github.io/pre-airdrop-detection/MINDMAP.html)

### Detection Horizon

How early can we detect, and does the signal degrade over time?

| Script | Purpose | Key result |
|--------|---------|------------|
| 01_build_features.py | Extract 18 behavioral features from raw transactions at each temporal cutoff, enforcing strict timestamp isolation | Preprocessing for all downstream experiments |
| 02_train_lightgbm.py | LightGBM 5-fold CV on T-30 data, compared against ARTEMIS | AUC 0.905 vs ARTEMIS 0.803 |
| 03_temporal_ablation.py | Sweep AUC from T-0 to T-90 in six windows | T-0: 0.908, T-30: 0.905, T-90: 0.902 |
| 12_extended_temporal.py | Extend the sweep to T-120, T-150, T-180 | T-180: 0.895, total drop over six months is 0.013 |

### Feature Analysis

Which of the 18 features carry signal, and does their importance shift across time windows?

| Script | Purpose | Key result |
|--------|---------|------------|
| 04_feature_importance.py | Track LightGBM gain-based importance at T-0, T-30, T-60, T-90 | NFT diversity stable at 38-49%; unique_interactions rises from 7% to 22% as window extends |
| 06_ablation_features.py | Remove one feature group at a time: Activity, Diversity, Volume, Behavioral, DeFi | DeFi alone achieves 0.544; sybils do not use Blend |
| 07_graph_features.py | Add graph structure features (address clustering, neighbor counts) to LightGBM | Behavioral 0.904, behavioral+graph 0.905; graph adds 0.001 |
| 10_shap_analysis.py | SHAP beeswarm on T-30 model | Confirms unique_interactions and buy_value as top contributors alongside diversity |
| 19_shap_temporal.py | Compare SHAP distributions at T-90 vs T-7 | T-90: wallet_age_days high (batch wallet creation signal); T-7: recent_activity high (sprint-before-distribution signal) |

### GNN Comparison

Why does the best existing method fail in the pre-airdrop setting?

| Script | Purpose | Key result |
|--------|---------|------------|
| 09_artemis_gnn.py | Reproduce ArtemisNet GNN with full post-hoc graph | AUC 0.976, confirms the post-hoc upper bound |
| 15_gnn_preairdrop.py | Apply ArtemisNet to pre-airdrop graph (T-30 cutoff) | AUC 0.586; collapses because airdrop transfer edges are absent |

### Generalization and Robustness

Does the model learn a real signal or overfit to Blur Season 2?

| Script | Purpose | Key result |
|--------|---------|------------|
| 05_pr_analysis.py | Full precision-recall curve for threshold selection at deployment | Documents trade-off across operating points |
| 08_generalization.py | Temporal split (train T-90, test T-30) and population split (50% unseen sybil types) | Temporal: 0.898; population: 0.744 |
| 11_sybil_clustering.py | K-means on sybil addresses to identify behavioral subtypes | K=3: 49K retail hunters, 601 mid-volume, 6 hyperactive bots (9K+ transactions each) |
| 20_clustering_k_selection.py | Elbow and silhouette analysis to justify K choice | Silhouette optimal at K=2 (0.984), K=3 chosen for semantic separation |
| 21_flag_type_generalization.py | Leave-one-flag-out: train without one sybil type, test on that type | BW=0.047, FD=0.110, ML=0.241, HF=0.550; each type has a distinct behavioral fingerprint |
| 22_calibration.py | Isotonic regression calibration of output probabilities | Brier score 0.120 to 0.091 (-24%), AUC unchanged |

### Adversarial Analysis

If a sybil hunter knows about the model, can they cheaply evade it?

| Script | Purpose | Key result |
|--------|---------|------------|
| 13_adversarial_robustness.py | Simulate diversity-reduction evasion at increasing levels | AUC drops slowly under evasion; 70% diversity reduction cuts AUC to 0.901 |
| 23_adversarial_cost.py | Quantify the economic cost of evasion in lost airdrop points | 90% diversity cut reduces AUC by 0.6% but costs 67% of expected points; evasion is self-defeating |

### Cross-Protocol Transfer

Does the method work on other protocols, and what determines transfer quality?

| Script | Purpose | Key result |
|--------|---------|------------|
| 14_cross_protocol.py | Blur-to-Hop zero-shot and fine-tune with protocol-specific features | Zero-shot AUC 0.500; 1% Hop labels recovers 0.982 |
| 16_lopo_crossprotocol.py | Leave-one-protocol-out across Blur, Hop, Gitcoin with protocol-specific features | AUC 0.22-0.47; initially looks like failure |
| 16b_lopo_blur_hop.py | Blur-Hop label-budget curve: how many Hop labels are needed? | 1% labels (roughly 300 samples) recovers AUC 0.982; 20% gives 0.981 |
| 17_common_features_lopo.py | Repeat LOPO with five protocol-agnostic features | Blur-to-Hop improves from 0.38 to 0.78; low LOPO AUC in Exp 16 was mostly feature mismatch |
| 18_gitcoin_feature_importance.py | Feature importance within the Gitcoin domain | gitcoin_donations importance = 0%; SADScore labels general on-chain anomalies, not Gitcoin-specific farming |
| 24_layerzero_lopo.py | Add LayerZero as fourth protocol; test same-class vs cross-domain transfer | Hop-to-LZ (bridge-to-bridge): 0.567; Blur-to-LZ (NFT-to-bridge): 0.434; protocol category affects transfer |

### Deployment Readiness

Can the remaining weaknesses be resolved before production use?

| Script | Purpose | Key result |
|--------|---------|------------|
| 25_openworld_detection.py | Two-stage detector: LightGBM for known types, Isolation Forest for unknown | IF raises BW detection from 0.047 to 0.916 without any BW labels |
| 26_label_noise.py | Flip 5-20% of labels randomly, retrain, measure AUC degradation | 20% label corruption drops AUC by 0.014; model is robust to imperfect blacklists |
| 27_rule_baselines.py | Compare against simple heuristics (wallet age, tx count) and weaker ML models | Best rule: 0.610; Logistic Regression: 0.859; Random Forest: 0.897; LightGBM: 0.905 |

## Reproducibility

All temporal cutoffs are enforced by filtering on raw transaction timestamps before any feature computation. Features at T-k use only transactions with timestamp less than (T0 minus k days in seconds). Labels are applied after feature computation and are never used to select which transactions to include. The Blur T0 is 1700525735 (first Season 2 claim transaction, 2023-11-21 00:15:35 UTC).
