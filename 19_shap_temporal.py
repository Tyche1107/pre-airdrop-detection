"""
19_shap_temporal.py
SHAP values at T-0, T-30, T-90 side by side to show how feature contributions shift over time.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

OUT_DIR = '/Users/adelinewen/Desktop/pre-airdrop-detection/data'
BLUR_T0 = 1700525735

WINDOWS = {
    'T-0':  'nft_feats_labeled_T7.csv',   # T-0 not saved separately, use T-7 as proxy
    'T-30': 'nft_feats_labeled_T30.csv',
    'T-90': 'nft_feats_labeled_T90.csv',
}

FEATURE_LABELS = {
    'buy_count': 'Buy Count',
    'sell_count': 'Sell Count',
    'tx_count': 'Total Tx Count',
    'total_trade_count': 'Total Trade Count',
    'buy_value': 'Buy Volume (ETH)',
    'sell_value': 'Sell Volume (ETH)',
    'pnl_proxy': 'PnL Proxy',
    'buy_collections': 'NFT Collection Diversity',
    'unique_interactions': 'Unique Interactions',
    'sell_ratio': 'Sell Ratio',
    'wallet_age_days': 'Wallet Age (days)',
    'days_since_last_buy': 'Days Since Last Buy',
    'recent_activity': 'Recent Activity',
}

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, (window_name, fname) in zip(axes, WINDOWS.items()):
    path = f'/Users/adelinewen/Desktop/pre-airdrop-detection/data/{fname}'
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ['address', 'is_sybil']]
    X = df[feature_cols].fillna(0).values.astype(float)
    y = df['is_sybil'].values

    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                            class_weight='balanced', random_state=42, verbose=-1)
    m.fit(X, y)

    explainer = shap.TreeExplainer(m)
    sample_idx = np.random.RandomState(42).choice(len(X), min(3000, len(X)), replace=False)
    X_sample = X[sample_idx]
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]

    mean_abs = np.abs(sv).mean(axis=0)
    imp_df = pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': mean_abs})
    imp_df = imp_df.sort_values('mean_abs_shap', ascending=True).tail(10)

    labels = [FEATURE_LABELS.get(f, f) for f in imp_df['feature']]
    bars = ax.barh(labels, imp_df['mean_abs_shap'], color='#7c6fcd', alpha=0.85)
    ax.set_title(f'{window_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean |SHAP value|', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)

plt.suptitle('SHAP Feature Importance Across Time Windows', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/shap_temporal_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved -> {OUT_DIR}/shap_temporal_comparison.png")
