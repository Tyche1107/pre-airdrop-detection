"""
22_calibration.py
Probability calibration analysis: is the model's predict_proba well-calibrated?
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/Users/adelinewen/Desktop/pre-airdrop-detection/data'

df = pd.read_csv(f'{OUT_DIR}/nft_feats_labeled_T30.csv')
FEATURES = [c for c in df.columns if c not in ['address', 'is_sybil']]
X = df[FEATURES].fillna(0).values.astype(float)
y = df['is_sybil'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Uncalibrated
raw = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                          class_weight='balanced', random_state=42, verbose=-1)
raw.fit(X_tr, y_tr)
prob_raw = raw.predict_proba(X_te)[:, 1]

# Calibrated (isotonic)
cal = CalibratedClassifierCV(
    lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                        class_weight='balanced', random_state=42, verbose=-1),
    cv=3, method='isotonic'
)
cal.fit(X_tr, y_tr)
prob_cal = cal.predict_proba(X_te)[:, 1]

# Calibration curves
frac_raw, mean_raw = calibration_curve(y_te, prob_raw, n_bins=15, strategy='uniform')
frac_cal, mean_cal = calibration_curve(y_te, prob_cal, n_bins=15, strategy='uniform')

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_raw, frac_raw, 's-', color='#7c6fcd', label=f'LightGBM (Brier={brier_score_loss(y_te, prob_raw):.4f})')
ax.plot(mean_cal, frac_cal, 'o-', color='#48bb78', label=f'+ Isotonic calibration (Brier={brier_score_loss(y_te, prob_cal):.4f})')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration Curve (T-30 model)')
ax.legend()
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/calibration_curve.png', dpi=150)
plt.close()

summary = pd.DataFrame({
    'model': ['LightGBM_raw', 'LightGBM_isotonic'],
    'auc': [roc_auc_score(y_te, prob_raw), roc_auc_score(y_te, prob_cal)],
    'brier_score': [brier_score_loss(y_te, prob_raw), brier_score_loss(y_te, prob_cal)]
})
summary.to_csv(f'{OUT_DIR}/calibration_results.csv', index=False)
print("Calibration results:")
print(summary.to_string(index=False))
print(f"Saved -> {OUT_DIR}/calibration_curve.png")
