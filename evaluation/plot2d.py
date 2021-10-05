import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot2d(x: np.ndarray, y, titles, dataset_name, centroids=None):
    """ the function aims to reduce the features to two dimensions using PCA method and plot the clusters
        x: 2D data array of size (rows, features).
        y: assigned labels.
        title: graphic title (example: iris dataset using KMeans).

    """
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    final_df = pd.DataFrame(data=principal_components,
                            columns=['principal component 1', 'principal component 2'])

    fig = plt.figure(figsize=(25, 25))
    for i in range(0, len(titles)):
        y_predict = y[i]
        ax = fig.add_subplot(4, 2, i + 1)
        ax.set_xlabel('Principal Component 1', fontsize=10)
        ax.set_ylabel('Principal Component 2', fontsize=10)
        ax.set_title('{}: {}'.format(titles[i], dataset_name), fontsize=15)
        if centroids is not None and titles[i] is not 'Original':
            pca_centroids = pca.transform(centroids[titles[i]])
            ax.scatter(pca_centroids[:, 0], pca_centroids[:, 1], c='red', s=100)
        targets = list(set(y_predict))
        y_predict = pd.DataFrame(y_predict).iloc[:, -1]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for target, color in zip(targets, colors):
            indices2keep = y_predict == target
            ax.scatter(final_df.loc[indices2keep, 'principal component 1'],
                       final_df.loc[indices2keep, 'principal component 2'],
                       c=color,
                       s=50)
        ax.grid()
    plt.savefig('results/{}.png'.format(dataset_name))
