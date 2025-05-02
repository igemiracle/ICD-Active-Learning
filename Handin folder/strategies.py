#!/usr/bin/env python
# coding: utf-8

# This file is used to define all kinds of active learning strategies for simple models and deep learning models. Functions here can be invoked to select the instances in active learning.


from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np



def diversity_sampling_kmeans(embeddings, batch_size=1, n_clusters=1, mode='random'):
    """
    embeddings: np.array shape (n_samples, n_features)
    mode: 'random' | 'centroid' | 'border'
    return: list of selected sample indices
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    clusters = kmeans.labels_
    centers = kmeans.cluster_centers_

    selected_idx = []

    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) == 0:
            continue

        if mode == 'random':
            idx = np.random.choice(cluster_indices, 1)[0]
        elif mode == 'centroid':
            cluster_points = embeddings[cluster_indices]
            dists = np.linalg.norm(cluster_points - centers[i], axis=1)
            idx = cluster_indices[np.argmin(dists)]
        elif mode == 'border':
            cluster_points = embeddings[cluster_indices]
            dists = np.linalg.norm(cluster_points - centers[i], axis=1)
            idx = cluster_indices[np.argmax(dists)]

        selected_idx.append(idx)

    # Compensate with random selected idx if # selected idx < batch size
    while len(selected_idx) < batch_size:
        candidates = list(set(range(len(embeddings))) - set(selected_idx))
        selected_idx.append(np.random.choice(candidates))

    return selected_idx[:batch_size]


def uncertainty_sampling(probabilities, batch_size=1, mode='least_confidence'):
    """
    probabilities: np.array shape (n_samples, n_classes), predicted probabilities
    mode: 'least_confidence' | 'binary_entropy' | 'smallest_margin'
    return: list of selected sample indices
    """
    if mode == 'least_confidence':
        scores = 1 - np.max(probabilities, axis=1)

    elif mode == 'binary_entropy':
        eps = 1e-12 
        scores = -np.sum(probabilities * np.log(probabilities + eps), axis=1)

    elif mode == 'smallest_margin':
        part = np.partition(-probabilities, 1, axis=1)
        top1 = -part[:, 0]
        top2 = -part[:, 1]
        scores = top2 - top1

    else:
        raise ValueError("Unsupported mode. Choose from 'least_confidence', 'binary_entropy', 'smallest_margin'.")

    # Select the top batch_size indexes with the highest scores
    selected_idx = np.argsort(scores)[-batch_size:][::-1]  # Descending order

    return selected_idx.tolist()




def information_density_sampling(embeddings, n_components=100, batch_size=10, random_state=42):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced_embeddings = svd.fit_transform(embeddings)
    similarity_matrix = cosine_similarity(reduced_embeddings)

    np.fill_diagonal(similarity_matrix, 0)

    density_scores = similarity_matrix.mean(axis=1)

    selected_idx = np.argsort(density_scores)[-batch_size:][::-1]

    return selected_idx.tolist()



