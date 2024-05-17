import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np


def tsne(feats, labels, imgname):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(feats)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    plt.figure(figsize=(8, 8))
    colors = plt.get_cmap('RdBu', 100)
    plt.scatter(X_norm[:, 0], X_norm[:, 1], labels, color=colors(labels))

    plt.xticks([])
    plt.yticks([])
    plt.savefig(imgname)
    plt.clf()
