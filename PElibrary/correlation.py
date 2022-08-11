import logging

import numpy as np


def kmeans(samples, k=None):
    from scipy.cluster import vq
    if k is None:
        from math import log
        k = max(1, int(round(log(len(samples), 3))))

    print("Clustering", len(samples), "samples by k-means, where k =", k)
    obs = pca_sk(samples, 3)
    obs = vq.whiten(obs)
    _centroids, labels = vq.kmeans2(obs, k, minit="++")

    from collections import defaultdict
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    clusters = clusters.values()

    return clusters


def markov(samples, inflation=5, max_iterations=100, by_pca=True):
    if inflation <= 1:
        raise ValueError("inflation must be > 1")

    if by_pca:
        pca_matrix = pca_sk(samples, 2)

        from scipy.spatial import distance
        dists = distance.squareform(distance.pdist(pca_matrix))
        M = 1 - (dists / dists.max())
    else:
        M = np.corrcoef(samples)

    M, clusters = mcl(M, max_iterations, inflation)

    return clusters


def mcl(M, max_iterations, inflation, expansion=2):
    print("M_init:\n", M)
    M = normalize(M)

    for i in range(max_iterations):
        M_prev = M
        M = inflate(expand(M, expansion), inflation)

        if converged(M, M_prev):
            logging.debug("Converged at iteration %d", i)
            break
        M = prune(M)

    clusters = get_clusters(M)
    return M, clusters


def normalize(A):
    return A / A.sum(axis=0)


def inflate(A, inflation):
    return normalize(np.power(A, inflation))


def expand(A, expansion):
    return np.linalg.matrix_power(A, expansion)


def converged(M, M_prev):
    return np.allclose(M, M_prev)


def get_clusters(M):
    attractors_idx = M.diagonal().nonzero()[0]
    clusters_idx = [M[idx].nonzero()[0]
                    for idx in attractors_idx]
    return clusters_idx


def prune(M, threshold=.001):
    pruned = M.copy()
    pruned[pruned < threshold] = 0
    return pruned


def pca_sk(data, n_components=None):
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components).fit_transform(data)


def pca_plain(data, n_components=None):
    data = data.copy()
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    C = np.cov(data)

    E, V = np.linalg.eigh(C)

    key = np.argsort(E)[::-1][:n_components]
    E, V = E[key], V[:, key]

    U = np.dot(data, V)

    return U


def plot_clusters(M, cluster_indices):
    from matplotlib import pyplot as plt

    _fig, ax = plt.subplots(1, 1)
    for cl_idx in cluster_indices:
        ax.scatter(M[cl_idx, 0], M[cl_idx, 1])
    plt.show()
