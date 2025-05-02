#!/usr/bin/env python
# coding: utf-8

# This file is used to define all kinds of active learning strategies for simple models and deep learning models. Functions here can be invoked to select the instances in active learning.


from sklearn.cluster import KMeans
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


def query_by_committee(comm_probs_list, batch_size=1, mode='vote_entropy'):
    """
    comm_probs_list: list of np.array, each of shape (n_samples, n_classes)
                     prediction probabilities from each committee member
    mode: 'vote_entropy' | 'kl_divergence'
    return: list of selected sample indices
    Note: This method needs at least 2 fitted models!
    """
    n_committee = len(comm_probs_list)
    n_samples = comm_probs_list[0].shape[0]
    n_classes = comm_probs_list[0].shape[1]

    if mode == 'vote_entropy':
        votes = np.array([np.argmax(p, axis=1) for p in comm_probs_list])  # shape: (n_committee, n_samples)
        vote_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_classes), axis=0, arr=votes)
        vote_distribution = vote_counts / n_committee  # shape: (n_classes, n_samples)
        vote_entropy = -np.sum(vote_distribution * np.log(vote_distribution + 1e-12), axis=0)  # shape: (n_samples,)
        scores = vote_entropy

    elif mode == 'kl_divergence':
        avg_probs = np.mean(np.stack(comm_probs_list, axis=0), axis=0)  # shape: (n_samples, n_classes)
        eps = 1e-12
        kl_divs = [np.sum(p * (np.log(p + eps) - np.log(avg_probs + eps)), axis=1) for p in comm_probs_list]
        mean_kl = np.mean(np.stack(kl_divs, axis=0), axis=0)  # shape: (n_samples,)
        scores = mean_kl

    else:
        raise ValueError("Unsupported mode. Choose from 'vote_entropy', 'kl_divergence'.")

    selected_idx = np.argsort(scores)[-batch_size:][::-1]

    return selected_idx.tolist()


def expected_model_change(clf, X_unlabeled_vec, batch_size=1):
    """
    Estimate expected model change for each sample (gradient-based), and return indices with highest change.
    Only works with linear models (e.g., LogisticRegression).
    
    Parameters:
    - clf: trained classifier (must support coef_)
    - X_unlabeled_vec: np.array or sparse matrix, shape (n_samples, n_features)
    - batch_size: number of samples to select

    Returns:
    - selected_idx: list of indices of selected samples
    """
    from scipy.special import softmax

    # Get raw decision scores (logits)
    logits = clf.decision_function(X_unlabeled_vec)  # shape: (n_samples, n_classes)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)

    probs = softmax(logits, axis=1)  # Convert to probability distribution

    n_samples, n_classes = probs.shape
    grad_norms = np.zeros(n_samples)

    for i in range(n_samples):
        x = X_unlabeled_vec[i]  # shape: (1, n_features), can be sparse

        # Compute expected gradient norm (weighted by class probabilities)
        expected_norm = 0
        for c in range(n_classes):
            p = probs[i, c]
            if hasattr(x, "toarray"):  # handle sparse
                x_dense = x.toarray().flatten()
            else:
                x_dense = x.flatten()

            grad = p * (1 - p) * x_dense  # ∇θ for logistic regression
            expected_norm += p * np.linalg.norm(grad)

        grad_norms[i] = expected_norm

    selected_idx = np.argsort(grad_norms)[-batch_size:][::-1]
    return selected_idx.tolist()


def iwal_sampling(clf, X_unlabeled_vec, batch_size=1, p_min=0.1):
    """
    IWAL sampling: query based on prediction uncertainty (max - min prob).
    Returns selected indices and their corresponding importance weights.
    """
    probs = clf.predict_proba(X_unlabeled_vec)  # shape: (n_samples, n_classes)
    if isinstance(probs, list):  # OneVsRestClassifier returns list of arrays
        probs = np.array(probs).transpose((1, 0))  # (n_samples, n_classes)

    # Compute disagreement-based sampling probability p_t
    p_t = np.clip(np.max(probs, axis=1) - np.min(probs, axis=1), p_min, 1.0)

    # Flip coin: decide whether to query label
    query_flags = np.random.rand(len(p_t)) < p_t
    queried_indices = np.where(query_flags)[0]

    # If queried set too small, supplement randomly
    if len(queried_indices) < batch_size:
        supplement = np.random.choice(list(set(range(len(p_t))) - set(queried_indices)),
                                      size=batch_size - len(queried_indices),
                                      replace=False)
        queried_indices = np.concatenate([queried_indices, supplement])

    # Importance weight = 1 / p_t for queried samples
    importance_weights = np.zeros(len(p_t))
    importance_weights[queried_indices] = 1.0 / p_t[queried_indices]

    # Select top-k highest p_t for label efficiency
    selected_idx = queried_indices[np.argsort(p_t[queried_indices])[-batch_size:]]

    return selected_idx.tolist(), importance_weights[selected_idx].tolist()

