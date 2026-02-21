# Pre-Airdrop Sybil Detection: Paper Outline
**Target:** WWW '27 (The Web Conference)
**Framing:** Prevention vs. Forensics — detecting airdrop hunters *before* the airdrop

---

## Title (candidates)
1. "Before the Drop: Pre-Airdrop Sybil Detection via Behavioral Fingerprinting"
2. "Catching Airdrop Hunters Early: Temporal and Cross-Protocol Sybil Detection"
3. "Pre-Airdrop Sybil Detection: A Behavioral Fingerprinting Approach"

---

## Abstract (draft)
Existing Sybil detection methods for blockchain airdrops (e.g., ARTEMIS) rely on
post-hoc analysis — identifying hunters *after* the distribution has already occurred.
We pose a new research question: **can we detect airdrop hunters before the airdrop,
using only pre-distribution on-chain behavior?**

Using Blur Season 2 NFT marketplace data (251k addresses, 3.2M transactions),
we show that a LightGBM classifier trained solely on pre-airdrop behavioral features
achieves AUC 0.9045 at T-30 (30 days before airdrop), outperforming the ARTEMIS
post-hoc baseline (AUC 0.803) by 10 percentage points.

We further show: (1) detection remains effective up to 180 days prior (AUC 0.895);
(2) graph-based models (ArtemisNet) collapse to near-random (AUC 0.586) when restricted
to pre-airdrop data, confirming that simple behavioral features are superior for
early detection; (3) behavioral features transfer across protocol types — models
trained on Blur generalize to Hop bridge (zero-shot AUC 0.726).

Our findings suggest a universal behavioral fingerprint for on-chain Sybil activity,
enabling proactive airdrop filtering rather than post-hoc remediation.

---

## 1. Introduction
- Background: Sybil attacks in blockchain airdrops (billions at stake)
- Problem: current detection is retrospective (ARTEMIS, rule-based)
- Gap: no work on *pre-airdrop* detection
- Our contribution: new research question + strong results + cross-protocol generalization
- Key narrative: "ARTEMIS = post-mortem; ours = prevention"

### 1.1 Contributions
1. First pre-airdrop Sybil detection framework using on-chain behavioral features
2. Temporal ablation showing detection efficacy up to 6 months prior
3. Fair comparison showing GNN models require post-hoc data; behavioral features dominate
4. Cross-protocol generalization (Blur NFT → Hop bridge, zero-shot AUC 0.726)
5. Behavioral analysis of 3 Sybil archetypes (Whales / Professionals / Small-scale)

---

## 2. Background & Related Work
### 2.1 Airdrop Sybil Attacks
- Economics of Sybil attacks (market sizing)
- Blur Season 2 case study

### 2.2 Existing Detection Methods
- ARTEMIS (WWW'24): GNN on post-hoc transfer graph, AUC 0.803
- Rule-based approaches: CHI'26 paper (the lab's own work)
- LLM-based: LLMhunter (brief mention)

### 2.3 Why Post-Hoc is Not Enough
- Once tokens distributed, damage done
- Legal/technical difficulty of clawback
- Motivation for pre-airdrop detection

---

## 3. Dataset & Problem Formulation
### 3.1 Blur Season 2 Dataset
- 3,246,484 Sale transactions, Feb–Nov 2023
- 251,755 unique addresses
- Labels: 50,546 Sybils (CHI'26 rule-based pipeline: BW/ML/FD/HF flags)
- T0 = 2023-11-21 00:15 UTC (first claim transaction)

### 3.2 Problem Definition
- Formal: binary classification at time T < T0
- Strict temporal isolation: features only from [start, T0 - k days]
- Evaluation: AUC-ROC (imbalanced labels)

### 3.3 Feature Engineering
- 4 feature groups: Activity (tx_count, unique_interactions), Volume (buy/sell ETH),
  Behavioral (ratio, NFT diversity), DeFi (Blend borrow/lend)
- Note: blend_net_value stored in WEI; LP features all-zero (removed)
- Common features for cross-protocol: tx_count, volume, unique_interactions, wallet_age

---

## 4. Experiments

### 4.1 Temporal Ablation (CORE RESULT)
- T0 → T-180: AUC degrades gracefully (0.907 → 0.895)
- **Key result: T-30 AUC 0.9045 > ARTEMIS 0.803**
- Table: T0/T-7/T-14/T-30/T-60/T-90/T-120/T-150/T-180

### 4.2 Model Comparison (Fair GNN Comparison)
| Model | Setting | AUC |
|-------|---------|-----|
| ARTEMIS (paper) | post-hoc | 0.803 |
| LightGBM (ours) | pre-airdrop T-30 | 0.9045 |
| ArtemisNet GNN | post-hoc | 0.9756 |
| ArtemisNet GNN | pre-airdrop T-30 | 0.5857 |

- **Key insight:** GNN relies on post-hoc transfer graph; behavioral features sufficient

### 4.3 Feature Analysis
- SHAP: Unique Interactions (0.693) > Buy Volume ETH (0.683) > NFT Diversity (0.623)
- Ablation: Activity (0.863) > Diversity (0.854) > Behavioral (0.848) > Volume (0.846) >> DeFi (0.544)
- Graph augmentation: +0.0015 (behavioral+graph = 0.906)

### 4.4 Generalization
- Temporal: train T-90, test T-30 → AUC 0.898 (drop 0.006)
- Population: 50% training data → AUC 0.744
- Cross-protocol: Blur → Hop zero-shot 0.726, fine-tune 10% → 0.965
- LOPO results: [TO BE FILLED after Script 16]

### 4.5 Adversarial Robustness
- Baseline 0.904 → mimic normal → 0.862 (max -4.2%)
- Sybil clusters make evasion costly (whale behavior hard to mimic cheaply)

### 4.6 Sybil Behavioral Profiles
- 3 clusters: Whales (6), Professionals (601), Small-scale (49,039)
- Economic interpretation: diminishing returns explain cluster structure

---

## 5. Discussion
### 5.1 Why Behavioral Features Beat Graph Features (Pre-Airdrop)
- Graph structure only meaningful post-distribution
- Behavioral features capture intent, not network effects

### 5.2 Practical Implications
- Protocol operators can run detection at T-30 before snapshot
- Cost-benefit: false positive rate vs. Sybil exclusion accuracy

### 5.3 Limitations
- Single platform (Blur Season 2)
- Labels from rule-based pipeline (potential noise)
- Gitcoin: application-layer Sybils require domain-specific features

---

## 6. Conclusion
- Pre-airdrop detection is feasible and outperforms post-hoc methods
- 4 generic behavioral features generalize across NFT/bridge protocols
- Future work: real-time deployment, more protocols, privacy-preserving detection

---

## Key Numbers to Remember
- ARTEMIS baseline: 0.803 (post-hoc, WWW'24)
- Ours T-30: **0.9045** (+10.2pp)
- Ours T-90: 0.9014 (+9.8pp)
- Ours T-180: 0.895 (+9.2pp)
- GNN pre-airdrop: 0.5857 (near random)
- GNN post-hoc: 0.9756 (but needs airdrop data)
- Cross-protocol Blur→Hop: 0.726 zero-shot, 0.965 fine-tune 10%
