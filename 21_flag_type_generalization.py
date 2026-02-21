"""
21_flag_type_generalization.py
Leave-one-flag-out generalization: train without one sybil flag type, test on that type.

Flag types derived from behavioral thresholds on sybil addresses:
  BW  (Bid Whale)         : buy_value  > 75th pct of sybil buy_value
  ML  (Market-loop)       : sell_ratio > 0.8 AND total_trade_count > median
  FD  (Farm-Dump)         : buy_count  > median AND pnl_proxy < 0
  HF  (High-Frequency)    : wallet_age_days < 25th pct AND buy_count > 75th pct

NOTE: airdrop_targets_behavior_flags.csv not found → flags derived from
      feature percentiles on the actual sybil population (real computed values).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/Users/adelinewen/Desktop/pre-airdrop-detection/data'
feats_path = f'{OUT_DIR}/nft_feats_labeled_T30.csv'

df = pd.read_csv(feats_path)
FEATURE_COLS = [c for c in df.columns if c not in ['address', 'is_sybil']]

sybils = df[df['is_sybil'] == 1].copy()
print(f"Total addresses: {len(df):,}  |  Sybils: {len(sybils):,}")

# ── Derive behavioral flag types from percentiles (all real computed values) ──
bv_75  = sybils['buy_value'].quantile(0.75)
tc_50  = sybils['total_trade_count'].quantile(0.50)
bc_50  = sybils['buy_count'].quantile(0.50)
bc_75  = sybils['buy_count'].quantile(0.75)
wa_25  = sybils['wallet_age_days'].quantile(0.25)

sybils = sybils.copy()
sybils['BW'] = (sybils['buy_value']         >  bv_75).astype(int)
sybils['ML'] = ((sybils['sell_ratio']        >  0.8) & (sybils['total_trade_count'] > tc_50)).astype(int)
sybils['FD'] = ((sybils['buy_count']         >  bc_50) & (sybils['pnl_proxy']       <  0)).astype(int)
sybils['HF'] = ((sybils['wallet_age_days']   <  wa_25) & (sybils['buy_count']       >  bc_75)).astype(int)

print("\nBehavioral flag thresholds (computed from sybil population):")
print(f"  BW: buy_value > {bv_75:.4f} ETH  →  {sybils['BW'].sum():,} sybils")
print(f"  ML: sell_ratio > 0.8 & trade_count > {tc_50:.0f}  →  {sybils['ML'].sum():,} sybils")
print(f"  FD: buy_count > {bc_50:.0f} & pnl_proxy < 0  →  {sybils['FD'].sum():,} sybils")
print(f"  HF: wallet_age_days < {wa_25:.1f} & buy_count > {bc_75:.0f}  →  {sybils['HF'].sum():,} sybils")

# Attach flags back to full df
flag_cols = ['BW', 'ML', 'FD', 'HF']
df = df.merge(sybils[['address'] + flag_cols], on='address', how='left')
for fc in flag_cols:
    df[fc] = df[fc].fillna(0).astype(int)

results = []
for flag in flag_cols:
    sybil_this_type = df[(df['is_sybil'] == 1) & (df[flag] == 1)]
    if len(sybil_this_type) < 20:
        print(f"Skipping {flag}: only {len(sybil_this_type)} sybils")
        continue

    # Train: all sybils EXCEPT this type + all normals
    test_sybil  = df[(df['is_sybil'] == 1) & (df[flag] == 1)]
    test_normal = df[df['is_sybil'] == 0].sample(
        min(len(test_sybil) * 4, len(df[df['is_sybil'] == 0])), random_state=42)
    test_df  = pd.concat([test_sybil, test_normal])
    train_mask = (df['is_sybil'] == 0) | ((df['is_sybil'] == 1) & (df[flag] == 0))
    train_df = df[train_mask & ~df.index.isin(test_df.index)]

    X_tr = train_df[FEATURE_COLS].fillna(0).values.astype(float)
    y_tr = train_df['is_sybil'].values
    X_te = test_df[FEATURE_COLS].fillna(0).values.astype(float)
    y_te = test_df['is_sybil'].values

    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                            class_weight='balanced', random_state=42, verbose=-1)
    m.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
    print(f"  Leave-{flag}-out: AUC {auc:.4f} (test n={len(test_sybil):,} sybils of type {flag})")
    results.append({'flag_left_out': flag, 'test_n_sybil': len(test_sybil), 'auc': round(auc, 4)})

if results:
    res_df = pd.DataFrame(results)
    res_df.to_csv(f'{OUT_DIR}/flag_type_generalization.csv', index=False)
    print("\nResults:")
    print(res_df.to_string(index=False))

    # Baseline AUC (from script 22 / standard T-30 model)
    BASELINE_AUC = 0.905  # placeholder from prior experiments

    plt.figure(figsize=(8, 4))
    plt.bar(res_df['flag_left_out'], res_df['auc'], color='#7c6fcd', alpha=0.85)
    plt.axhline(BASELINE_AUC, color='gray', linestyle='--', label=f'Standard T-30 AUC={BASELINE_AUC}')
    plt.xlabel('Sybil Flag Type Left Out')
    plt.ylabel('AUC')
    plt.title('Leave-One-Flag-Out Generalization')
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/flag_type_generalization.png', dpi=150)
    plt.close()
    print(f"Saved -> {OUT_DIR}/flag_type_generalization.png")
else:
    print("No results to save — all flag types had too few sybils.")
