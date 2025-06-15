import numpy as np

def kmeans_plusplus(graph, k):
    n = len(graph.nodes)
    centroids = []
    centroids.append(np.random.randint(n))

    for _ in range(k - 1):
        distances = np.min(
            graph.adj_mat[:, centroids], axis=1
        )
        next_centroid = np.argmax(distances)
        centroids.append(next_centroid)
    
    return np.array(centroids)