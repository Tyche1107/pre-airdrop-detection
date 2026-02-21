"""
20_clustering_k_selection.py
Elbow curve + silhouette scores to justify K=3 for Sybil clustering.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/Users/adelinewen/Desktop/pre-airdrop-detection/data'

df = pd.read_csv(f'{OUT_DIR}/nft_feats_labeled_T30.csv')
sybil = df[df['is_sybil'] == 1].copy()

FEATURES = ['buy_count', 'sell_count', 'buy_value', 'sell_value',
            'buy_collections', 'unique_interactions', 'wallet_age_days']
X = sybil[FEATURES].fillna(0).values.astype(float)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

inertias, silhouettes = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    inertias.append(km.inertia_)
    if k > 1:
        # Use full data (n_sybil ~10k, manageable); skip sample_size to avoid 1-label edge case
        sil = silhouette_score(Xs, labels)
        silhouettes.append(sil)
    else:
        silhouettes.append(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_range), inertias, 'o-', color='#7c6fcd')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (Within-cluster SSE)')
ax1.set_title('Elbow Curve')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.plot(list(K_range), silhouettes, 's-', color='#48bb78')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score by K')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/clustering_k_selection.png', dpi=150)
plt.close()

results = pd.DataFrame({'k': list(K_range), 'inertia': inertias, 'silhouette': silhouettes})
results.to_csv(f'{OUT_DIR}/clustering_k_selection.csv', index=False)
print("K selection results:")
print(results.to_string(index=False))
print(f"\nBest K by silhouette: {results.loc[results.silhouette.idxmax(), 'k']}")
print(f"Saved -> {OUT_DIR}/clustering_k_selection.png")
