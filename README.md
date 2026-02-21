# Pre-Airdrop Sybil Detection

Existing work on airdrop hunter detection (ARTEMIS, WWW'24) operates retrospectively: it identifies Sybil addresses after the airdrop has been distributed, when the damage is already done. This project asks whether detection is possible before distribution, using only on-chain behavioral signals available prior to the airdrop announcement.

We frame this as a binary classification problem and study the earliest detection horizon at which a classifier achieves reliable performance.

## Key Results

LightGBM trained on pre-airdrop behavioral features achieves AUC 0.905 at T-30 (one month before distribution), outperforming ARTEMIS (AUC 0.803) despite using strictly pre-airdrop data. Performance degrades gracefully: AUC 0.895 at T-180, six months before the event.

The most predictive feature is NFT collection diversity. Sybil addresses spread activity across many collections to maximize points, a strategy visible in their transaction history well before the airdrop.

Cross-protocol zero-shot transfer performs poorly (AUC 0.22-0.47), but fine-tuning with 1% of target-protocol labels recovers near-perfect performance (AUC 0.98), suggesting protocol-specific adaptation requires minimal labeled data.

## Dataset

Blur Season 2 on-chain transactions (Feb-Nov 2023, 3.2M transactions, 251K addresses). Labels: BW / ML / FD / HF flags from `airdrop_targets_behavior_flags.csv`. Cross-protocol evaluation uses Hop Protocol, Gitcoin GR15, and LayerZero ZRO airdrop data.

## Experiments

| Script | Description |
|--------|-------------|
| 01_build_features.py | Extract 18 behavioral features at each temporal cutoff |
| 02_train_lightgbm.py | LightGBM baseline, 5-fold CV |
| 03_temporal_ablation.py | AUC across T-0 to T-90 cutoffs |
| 04_feature_importance.py | Feature importance stability across time windows |
| 05_pr_analysis.py | Precision-recall curve and threshold analysis |
| 06_ablation_features.py | Feature group ablation (Activity / Diversity / Volume / Behavioral / DeFi) |
| 07_graph_features.py | Graph-augmented features added to LightGBM |
| 08_generalization.py | Temporal and population generalization experiments |
| 09_artemis_gnn.py | ArtemisNet GNN reproduction (post-hoc baseline) |
| 10_shap_analysis.py | SHAP feature attribution |
| 11_sybil_clustering.py | Unsupervised clustering of Sybil subtypes |
| 12_extended_temporal.py | Extended temporal ablation to T-180 |
| 13_adversarial_robustness.py | Robustness under diversity-reduction evasion |
| 14_cross_protocol.py | Blur to Hop zero-shot and fine-tuned transfer |
| 15_gnn_preairdrop.py | ArtemisNet applied to pre-airdrop data (fair comparison) |
| 16_lopo_crossprotocol.py | Leave-one-protocol-out evaluation: Blur / Hop / Gitcoin |
| 16b_lopo_blur_hop.py | Blur-Hop transfer with varying label budget (1% to 20%) |

Full experiment map with results and figures: [MINDMAP.html](https://tyche1107.github.io/pre-airdrop-detection/MINDMAP.html)

## Technical Notes

Temporal isolation is enforced strictly: all features are computed from transactions with timestamp less than the cutoff date. Labels are post-hoc observations of a pre-existing behavioral identity and do not constitute leakage.

ArtemisNet GNN achieves AUC 0.976 in the post-hoc setting but collapses to 0.586 under pre-airdrop conditions, because its graph structure depends on transfer events generated at airdrop distribution. LightGBM on behavioral features does not have this dependency.
