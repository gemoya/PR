import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap


def plot2d(X,color):
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:,1], marker='o', c=color)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Cartesian coordinate PCA: ')
    cbar = plt.colorbar()
    cbar.set_label('?')
    plt.show()



X,color = make_swiss_roll(n_samples=3000, noise=0.05, random_state=None)
n_components = 2

# PCA
pca = PCA(n_components)
X_pca = pca.fit_transform(X)

# KernelPCA
kpca = KernelPCA(n_components, kernel="poly", degree=2)
X_kpca = kpca.fit_transform(X)

# IsoMap
#Y = Isomap(n_neighbors=10, n_components).fit_transform(X)


## plot
print color
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:,1], marker='o', c=color)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Cartesian coordinate PCA: ')
cbar = plt.colorbar()
cbar.set_label('?')
plt.show()




