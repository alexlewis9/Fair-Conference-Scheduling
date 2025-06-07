import os

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def get_pairwise_distances(clustering, distance_matrix, same_cluster=True):
    n = len(clustering)
    pairs = combinations(range(n), 2)
    distances = [
        distance_matrix[i][j]
        for i, j in pairs
        if (clustering[i] == clustering[j]) == same_cluster and clustering[i] != -1 and clustering[j] != -1
    ]
    return sorted(distances)

# Plotting
def plot_cluster_distances(clusterings, graph, path = '',  same_cluster=True, title=''):
    distance_matrix = graph.adj_mat
    plt.figure(figsize=(10, 6))
    for method, labels in clusterings.items():
        labels = graph.flatten_clusters(labels)
        dists = get_pairwise_distances(labels, distance_matrix, same_cluster=same_cluster)
        plt.plot(range(len(dists)), dists, label=method)
    plt.xlabel('Pair index (sorted)')
    plt.ylabel('Distance')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    path = os.path.join(path, f'{title}.jpeg')
    plt.savefig(path, format='jpeg', dpi=300)
    plt.close()


