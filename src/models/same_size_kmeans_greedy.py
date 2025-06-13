"""
Same-size k-Means Greedy Implementation
"""
import numpy as np
import heapq
from collections import defaultdict
from scipy.spatial.distance import cdist
from src.models.kmeans_plusplus import kmeans_plusplus

def same_size_kmeans_greedy(graph, max_iter=100):
    n = len(graph.nodes)
    k = graph.k

    size_floor = n // k
    size_ceil = size_floor + 1
    num_ceil = n % k
    cluster_sizes = [size_ceil]*num_ceil + [size_floor]*(k-num_ceil)

    # Initialization
    centers = kmeans_plusplus(graph, k)
    assigned = np.full(n, -1)
    cluster_counts = [0]*k

    dists = graph.adj_mat[:, centers]
    nearest = np.argsort(dists, axis=1)
    heap = []
    for i in range(n):
        heapq.heappush(heap, (dists[i, nearest[i,0]], i, 0, nearest[i].tolist()))

    while heap:
        dist, i, rank, order = heapq.heappop(heap)
        for r in range(rank, k):
            c = order[r]
            if cluster_counts[c] < cluster_sizes[c]:
                assigned[i] = c
                cluster_counts[c] += 1
                break

    # Iteration step
    for it in range(max_iter):
        # E-step
        centers_emb = np.array([
            graph.embeddings[assigned == c].mean(axis=0) if np.any(assigned == c) else graph.embeddings[centers[c]]
            for c in range(k)
        ])
        dists = cdist(graph.embeddings, centers_emb, metric=graph.d)

        # M-step
        proposals = defaultdict(list)
        moved = False
        for i in range(n):
            current = assigned[i]
            # Find clusters closer than current
            closer_clusters = np.where(dists[i] < dists[i, current])[0]
            for c in closer_clusters:
                if cluster_counts[c] < cluster_sizes[c]:
                    # Move to smaller cluster
                    cluster_counts[current] -= 1
                    cluster_counts[c] += 1
                    assigned[i] = c
                    moved = True
                    break
                else:
                    # Propose swap
                    proposals[c].append(i)
        # Try swaps
        for c, indices in proposals.items():
            for i in indices:
                current = assigned[i]
                # Find candidate in c to swap with
                candidates = np.where(assigned == c)[0]
                best_j = None
                best_improve = 0
                for j in candidates:
                    gain = (dists[i, c] + dists[j, current]) - (dists[i, current] + dists[j, c])
                    if gain < best_improve:
                        best_improve = gain
                        best_j = j
                if best_j is not None:
                    # Swap assignments
                    assigned[i], assigned[best_j] = assigned[best_j], assigned[i]
                    moved = True
        if not moved:
            break

    return assigned