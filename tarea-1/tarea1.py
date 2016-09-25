import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll

X_sr,color = make_swiss_roll(n_samples=3000, noise=0.05, random_state=None)
n_components = 2
pca = PCA(n_components)
X_sr2 = pca.fit_transform(X)