# Pre-Airdrop Sybil Detection

## Research Question
Can we detect airdrop hunter addresses *before* the airdrop is distributed,
using only on-chain behavioral features available prior to the airdrop announcement?

Existing work (ARTEMIS, WWW'24) identifies hunters retrospectively — after the
airdrop has been distributed. This project explores a predictive framing:
given behavior signals up to T days before the airdrop, can we classify which
addresses will exhibit Sybil behavior?

**Core question:** What is the earliest detection horizon — the minimum number
of days before the airdrop — at which a classifier achieves sufficient confidence?

## Dataset
Blur Season 2 airdrop data. Labels from `database/airdrop_targets_behavior_flags.csv`
(BW / ML / FD / HF flags). Features extracted from pre-airdrop on-chain activity.

## Approach
- Phase 1: Tabular features + LightGBM baseline
- Phase 2: Graph-based features + GraphSAGE
- Key experiment: temporal ablation (coarse: T-7/30/60/90, fine: scan around inflection)

## Key Technical Challenge
Strict temporal isolation: all features must be computable from data available
before the airdrop announcement. Labels (flags) represent address identity/behavior
patterns — they are post-hoc observations of a pre-existing identity, not leakage.
