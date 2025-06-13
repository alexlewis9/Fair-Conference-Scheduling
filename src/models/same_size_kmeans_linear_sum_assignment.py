""""
Same-size k-Means using linear sum assignment
"""
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def same_size_kmeans_linear_sum_assignment(graph, k):
    X = graph.embeddings
    cluster_size = X.shape[0] // k
    kmeans = KMeans(k)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
    distance_matrix = cdist(X, centers)
    clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
    return clusters