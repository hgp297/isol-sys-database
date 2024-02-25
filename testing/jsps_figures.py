import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import lognorm
from math import log, exp

from scipy.stats import norm
inv_norm = norm.ppf(0.84)
beta_drift = 0.25
mean_log_drift = exp(log(0.03) - beta_drift*inv_norm) # 0.9945 is inverse normCDF of 0.84
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

label_size = 16
clabel_size = 12
x = np.linspace(0, 0.15, 200)

mu = log(0.03)- 0.25*inv_norm
sigma = 0.25

ln_dist = lognorm(s=sigma, scale=exp(mu))
p = ln_dist.cdf(np.array(x))

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(8,6))

ax.plot(x, p, label='Collapse (peak)', color='blue')

mu_irr = log(0.01)
ln_dist_irr = lognorm(s=0.3, scale=exp(mu_irr))
p_irr = ln_dist_irr.cdf(np.array(x))

ax.plot(x, p_irr, color='red', label='Irreparable (residual)')

axis_font = 20
subt_font = 18
xright = 0.0
xleft = 0.15
ax.set_ylim([0,1])
ax.set_xlim([0, 0.05])
ax.set_ylabel('Damage probability', fontsize=axis_font)
ax.set_xlabel('Peak drift ratio', fontsize=axis_font)

# ax.vlines(x=exp(mu), ymin=0, ymax=0.5, color='blue', linestyle=":")
# ax.hlines(y=0.5, xmin=xright, xmax=exp(mu), color='blue', linestyle=":")
# ax.text(0.01, 0.52, r'$\theta = 0.078$', fontsize=axis_font, color='blue')
# ax.plot([exp(mu)], [0.5], marker='*', markersize=15, color="blue", linestyle=":")

# ax.vlines(x=0.1, ymin=0, ymax=0.84, color='blue', linestyle=":")
# ax.hlines(y=0.84, xmin=xright, xmax=0.1, color='blue', linestyle=":")
# ax.text(0.01, 0.87, r'$\theta = 0.10$', fontsize=axis_font, color='blue')
# ax.plot([0.10], [0.84], marker='*', markersize=15, color="blue", linestyle=":")

# lower= ln_dist.ppf(0.16)
# ax.vlines(x=lower, ymin=0, ymax=0.16, color='blue', linestyle=":")
# ax.hlines(y=0.16, xmin=xright, xmax=lower, color='blue', linestyle=":")
# ax.text(0.01, 0.19, r'$\theta = 0.061$', fontsize=axis_font, color='blue')
# ax.plot([lower], [0.16], marker='*', markersize=15, color="blue", linestyle=":")


ax.set_title('Component fragility function', fontsize=axis_font)
# ax.legend(fontsize=label_size, loc='upper center')
plt.show()
#%% sample clustering

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

fig, ax = plt.subplots(1, 1, figsize=(8,6))
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Sample building classification")

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=8,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=8,
    )
plt.text(-1.3, 1.8, 'Ductile steel frame', fontsize=14)
plt.text(0, -1.8, 'Non-ductile steel frame', fontsize=14)
plt.text(-2, 0.5, 'Concrete frame', fontsize=14)
plt.title(f"Sample building classification")
ax.axis('off')
plt.show()
