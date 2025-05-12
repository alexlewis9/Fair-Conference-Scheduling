import heapq
import numpy as np

def smallest_agent_ball(embeddings, adj_matrix, threshold):
    """
    Finds the smallest ball centered at an agent that captures at least `threshold` agents.

    Args:
        embeddings (np.ndarray): Embedding matrix where each row corresponds to a node's embedding.
        adj_matrix (np.ndarray): Adjacency matrix storing pairwise distances.
        threshold (int): Minimum number of agents to include in the ball.

    Returns:
        tuple: (center_index, cluster_indices) where:
            - center_index is the index of the center of the smallest ball.
            - cluster_indices is the list of indices of agents in the ball.
    """
    n = len(embeddings)
    min_radius = float('inf')
    best_center = None
    best_cluster = None

    for i in range(n):
        # Get distances from point i to all other points
        distances = adj_matrix[i]
        # Find the threshold-th smallest distance (radius of the ball)
        kth_smallest_distance = np.partition(distances, threshold - 1)[threshold - 1]
        # Get the indices of the points within the ball
        cluster_indices = np.where(distances <= kth_smallest_distance)[0]

        # Update if this ball is smaller
        if kth_smallest_distance < min_radius:
            min_radius = kth_smallest_distance
            best_center = i
            best_cluster = cluster_indices

    return best_center, best_cluster


def greedy_cohesive_clustering(embeddings, adj_matrix, k):
    """
    Greedy algorithm to cluster agents into k cohesive clusters.

    Args:
        embeddings (np.ndarray): Embedding matrix where each row corresponds to a node's embedding.
        adj_matrix (np.ndarray): Adjacency matrix storing pairwise distances.
        k (int): Number of clusters.

    Returns:
        list: A list of k clusters, where each cluster is a list of node indices.
    """
    n = len(embeddings)
    threshold = n // k  # Minimum number of agents per cluster
    remaining_indices = set(range(n))
    clusters = []

    while len(remaining_indices) >= threshold:
        # Create a submatrix for the remaining points
        remaining_indices_list = list(remaining_indices)
        submatrix = adj_matrix[np.ix_(remaining_indices_list, remaining_indices_list)]

        # Find the smallest ball in the remaining points
        center, cluster = smallest_agent_ball(embeddings[remaining_indices_list], submatrix, threshold)

        # Map cluster indices back to the original indices
        cluster_original_indices = [remaining_indices_list[i] for i in cluster]

        # Add the cluster to the result
        clusters.append(cluster_original_indices)

        # Remove the clustered points from the remaining set
        remaining_indices -= set(cluster_original_indices)

    # Add empty clusters if fewer than k clusters were created
    while len(clusters) < k:
        clusters.append([])

    return clusters