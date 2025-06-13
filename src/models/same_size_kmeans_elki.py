import numpy as np
from src.models.kmeans_plusplus import kmeans_plusplus

def same_size_kmeans_elki(graph, k, max_iter=100):
    # Initialization
    n = len(graph.nodes)
    cluster_size = n // k
    remainder = n % k
    centers = kmeans_plusplus(graph, k)

    distances = graph.adj_mat[:, centers]

    nearest = np.argmin(distances, axis=1)
    farthest = np.argmax(distances, axis=1)
    benefit = distances[np.arange(n), nearest] - distances[np.arange(n), farthest]
    order = np.argsort(benefit)

    clusters = [[] for _ in range(k)]
    assignments = np.full(n, -1)
    cluster_counts = [0] * k

    for i in order:
        for c in np.argsort(distances[i]):
            if cluster_counts[c] < cluster_size + (1 if c < remainder else 0):
                clusters[c].append(i)
                assignments[i] = c
                cluster_counts[c] += 1
                break

    # Iteration
    for iteration in range(max_iter):
        # Compute current cluster means
        means = []
        for c in range(k):
            if clusters[c]:
                means.append(np.mean(graph.adj_mat[clusters[c]], axis=0))
            else:
                means.append(np.zeros(graph.adj_mat.shape[1]))
        means = np.array(means)

        # Compute distances to cluster means
        dists = np.zeros((n, k))
        for i in range(n):
            for c in range(k):
                dists[i, c] = np.linalg.norm(graph.adj_mat[i] - means[c])

        # Sort by delta of curr assignment and best alternate assignment
        current_dists = dists[np.arange(n), assignments]
        best_alt_dists = np.partition(dists, 1, axis=1)[:, 1]
        delta = current_dists - best_alt_dists
        priority = np.argsort(-delta)

        moved = set()

        for i in priority:
            if i in moved:
                continue
            curr_c = assignments[i]
            gains = dists[i] - current_dists[i]
            for alt_c in np.argsort(gains):
                if alt_c == curr_c:
                    continue
                candidates = [j for j in clusters[alt_c] if j not in moved]
                # Try swap elements
                if candidates:
                    swap_j = candidates[0]
                    gain_i = dists[i, curr_c] - dists[i, alt_c]
                    gain_j = dists[swap_j, alt_c] - dists[swap_j, curr_c]
                    if gain_i + gain_j > 0:
                        clusters[curr_c].remove(i)
                        clusters[alt_c].append(i)
                        assignments[i] = alt_c

                        clusters[alt_c].remove(swap_j)
                        clusters[curr_c].append(swap_j)
                        assignments[swap_j] = curr_c

                        moved.add(i)
                        moved.add(swap_j)
                        break
                # Try moving without violating size constraints
                if cluster_counts[alt_c] < cluster_size + (1 if alt_c < remainder else 0):
                    clusters[curr_c].remove(i)
                    clusters[alt_c].append(i)

    final_clusters = [np.array(cluster) for cluster in clusters]
    return final_clusters, assignments