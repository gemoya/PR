import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap
from sklearn.lda import LDA
from scipy.io import loadmat
from tempfile import TemporaryFile

X_sr,colors_sr = make_swiss_roll(n_samples=3000, noise=0.05, random_state=None)
n_components = 2
swissroll = TemporaryFile()
np.savez(swissroll, X= X_sr, color=colors_sr)
swissroll.seek(0)
