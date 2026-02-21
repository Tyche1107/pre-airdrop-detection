"""
23_adversarial_cost.py
Quantify the real cost of evasion: if a sybil reduces diversity by X%,
how much do their Blur points (estimated from buy_collections) drop?
Shows the self-defeating nature of diversity reduction evasion.

Uses 5-fold CV so AUC values are consistent with the main results (exp 02).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/Users/adelinewen/Desktop/pre-airdrop-detection/data'

df = pd.read_csv(f'{OUT_DIR}/nft_feats_labeled_T30.csv')
FEATURES = [c for c in df.columns if c not in ['address', 'is_sybil']]

X_orig = df[FEATURES].fillna(0).values.astype(float)
y = df['is_sybil'].values
col_idx = FEATURES.index('buy_collections')

# Blur points proxy: proportional to buy_count * sqrt(buy_collections)
sybil_mask = y == 1
buy_count_col = FEATURES.index('buy_count')
baseline_points_vec = X_orig[sybil_mask, buy_count_col] * np.sqrt(
    np.clip(X_orig[sybil_mask, col_idx], 1, None))
baseline_points = baseline_points_vec.mean()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
reduction_levels = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

# For each reduction level, collect OOF predictions
oof_scores = {r: [] for r in reduction_levels}

for fold, (train_idx, test_idx) in enumerate(skf.split(X_orig, y)):
    X_train, y_train = X_orig[train_idx], y[train_idx]
    X_test_orig, y_test = X_orig[test_idx], y[test_idx]

    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                            class_weight='balanced', random_state=42, verbose=-1)
    m.fit(X_train, y_train)

    for r in reduction_levels:
        X_test_mod = X_test_orig.copy()
        X_test_mod[:, col_idx] = X_test_orig[:, col_idx] * (1 - r)
        auc = roc_auc_score(y_test, m.predict_proba(X_test_mod)[:, 1])
        oof_scores[r].append(auc)

baseline_auc = np.mean(oof_scores[0])
print(f"Baseline AUC (5-fold CV): {baseline_auc:.4f}")

results = []
for r in reduction_levels:
    auc = np.mean(oof_scores[r])
    detection_pct = auc / baseline_auc * 100

    # Points also drop (simple formula, same for all folds)
    mod_collections = np.clip(X_orig[sybil_mask, col_idx] * (1 - r), 1, None)
    mod_points = (X_orig[sybil_mask, buy_count_col] * np.sqrt(mod_collections)).mean()
    points_pct = mod_points / baseline_points * 100

    results.append({'diversity_reduction_pct': int(r * 100),
                    'auc': round(auc, 4),
                    'auc_vs_baseline_pct': round(detection_pct, 1),
                    'points_vs_baseline_pct': round(points_pct, 1)})
    print(f"  Diversity -{r*100:.0f}%: AUC={auc:.4f} | Points retained={points_pct:.1f}%")

res_df = pd.DataFrame(results)
res_df.to_csv(f'{OUT_DIR}/adversarial_cost.csv', index=False)

fig, ax1 = plt.subplots(figsize=(9, 5))
x = res_df['diversity_reduction_pct']
ax1.plot(x, res_df['auc'], 'o-', color='#7c6fcd', linewidth=2, label='Model AUC (evasion benefit)')
ax1.set_xlabel('Diversity Reduction (%)')
ax1.set_ylabel('Model AUC', color='#7c6fcd')
ax1.tick_params(axis='y', labelcolor='#7c6fcd')
ax1.set_ylim(0.5, 1.0)

ax2 = ax1.twinx()
ax2.plot(x, res_df['points_vs_baseline_pct'], 's--', color='#fc8181', linewidth=2, label='Estimated Blur Points (%)')
ax2.set_ylabel('Blur Points Retained (%)', color='#fc8181')
ax2.tick_params(axis='y', labelcolor='#fc8181')
ax2.set_ylim(0, 110)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
ax1.set_title('Evasion vs Cost: Diversity Reduction Trade-off', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/adversarial_cost_tradeoff.png', dpi=150)
plt.close()
print(f"Saved -> {OUT_DIR}/adversarial_cost_tradeoff.png")
print(res_df.to_string(index=False))
