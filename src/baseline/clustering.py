import cPickle as p
import matplotlib.pyplot as plt
import os.path as o
import time

import numpy as np
import seaborn as sns
from sklearn import cluster
from sklearn.decomposition import PCA

sns.set_context('poster')
sns.set_color_codes()
current_palette = sns.color_palette("hls", 22)
sns.set_palette(current_palette)
plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

root = o.abspath(o.dirname(__file__))

np.random.seed(1234)
random_state = 9
DUMP = o.join(root, '../../data/formatted_data.p')
LDA_DUMP = o.join(root, '../../data/lda_dump.p')
LDA_MODEL = o.join(root, '../../data/lda_model.p')


def plot_clusters(data, algorithm, args, kwds):
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    labels = algorithm(*args, **kwds).fit_predict(data)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=16)
    plt.show()


if __name__ == '__main__':
    X = p.load(open(LDA_DUMP, 'rb'))
    plot_clusters(X, cluster.KMeans, (), {'n_clusters': 22})
    # plot_clusters(X, cluster.AgglomerativeClustering, (), {'n_clusters': 6, 'linkage': 'ward'})
    # plot_clusters(X, cluster.DBSCAN, (), {'eps': 0.025})
