"""
18_gitcoin_feature_importance.py
Explain why Gitcoin in-domain AUC is 0.634.
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

onchain = pd.read_csv('/Users/adelinewen/Desktop/dataset/gitcoin/onchain_features.csv')
erc20   = pd.read_csv('/Users/adelinewen/Desktop/dataset/gitcoin/erc20_features.csv')
df = onchain.merge(erc20[['address','token_tx_count','stable_volume_usd','unique_token_contracts','gitcoin_donations']],
                   on='address', how='left').fillna(0)

FEATURES = ['tx_count','eth_sent','eth_received','total_volume','unique_contracts',
            'wallet_age_days','token_tx_count','stable_volume_usd','unique_token_contracts','gitcoin_donations']
X = df[FEATURES].values.astype(float)
y = df['is_sybil'].values

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for tr,te in kf.split(X,y):
    m = lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=31,
                            class_weight='balanced',random_state=42,verbose=-1)
    m.fit(X[tr],y[tr])
    aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:,1]))
print(f"Gitcoin AUC (10 features): {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

m_full = lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=31,
                              class_weight='balanced',random_state=42,verbose=-1)
m_full.fit(X,y)

imp = pd.DataFrame({'feature':FEATURES,'importance':m_full.feature_importances_})
imp['pct'] = imp['importance']/imp['importance'].sum()*100
imp = imp.sort_values('importance',ascending=False)
print(imp.to_string(index=False))
imp.to_csv(f'{OUT_DIR}/gitcoin_feature_importance.csv', index=False)

explainer = shap.TreeExplainer(m_full)
idx = np.random.RandomState(42).choice(len(X), min(3000,len(X)), replace=False)
sv = explainer.shap_values(X[idx])
if isinstance(sv,list): sv = sv[1]

plt.figure(figsize=(10,6))
shap.summary_plot(sv, pd.DataFrame(X[idx],columns=FEATURES), show=False, plot_type='bar')
plt.title('Gitcoin SHAP Feature Importance')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/gitcoin_shap.png', dpi=150)
plt.close()

comp = pd.DataFrame(X,columns=FEATURES)
comp['is_sybil'] = y
comp.groupby('is_sybil')[FEATURES].mean().T.to_csv(f'{OUT_DIR}/gitcoin_sybil_vs_normal.csv')
print("Sybil vs Normal means saved.")
print(f"Saved -> {OUT_DIR}/gitcoin_shap.png")
