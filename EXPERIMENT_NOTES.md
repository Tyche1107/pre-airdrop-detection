# Experiment Design Notes

## Exp 16 vs Exp 17
Exp 16 used protocol-specific features — poor AUC (0.22-0.47) reflects feature mismatch, not behavioral non-transferability. Exp 17 uses 5 common features across protocols, improving to 0.624-0.780. These are two different models answering different questions.

## BW Type and IF (Exp 21 / 25)
IF recovers BW detection (0.916) because BW addresses are statistical outliers in buy_value. This is Blur-specific. For new protocols, IF effectiveness requires per-deployment validation.

## Sybil Type Nomenclature
Two systems exist: (1) BW/FD/ML/HF = external Blur ground-truth labels; (2) Retail/Mid/Hyperactive = K-means clusters from Exp 19. Different analyses, not conflicting.

## T-0 vs T-30 AUC
T-0 (0.9068) > T-30 (0.9045): sybils exhibit last-mile concentrated activity before the snapshot date, creating stronger signal at T-0. Not data leakage — the near-flat curve across all cutoffs is the primary finding.
