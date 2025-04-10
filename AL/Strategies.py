#!/usr/bin/env python
# coding: utf-8

# This file is used to define all kinds of active learning strategies for simple models and deep learning models. Functions here can be invoked to select the instances in active learning.

# In[2]:


from sklearn.cluster import KMeans
import numpy as np


# In[4]:


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


# In[ ]:




