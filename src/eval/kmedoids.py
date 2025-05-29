import numpy as np
from scipy.spatial.distance import cdist


def kmedoid_objective(graph, clustering, metric=None):
    """
    Computes the K-Medoids objective cost for a given clustering.

    Args:
        clustering (list[list[int]]): A list of clusters, where each cluster is a list of node IDs.
        metric (str, optional): Distance metric to use. Defaults to the graph's metric.

    Returns:
        float: Total K-Medoids objective cost.
    """
    metric = metric if metric else graph.d
    # Calculate new adj_mat if use different metric
    D = graph.adj_matrix if metric == graph.d else cdist(graph.embeddings, graph.embeddings, metric=metric)
    total_cost = 0

    for cluster in clustering:
        indices = [graph.id_to_index[nid] for nid in cluster]
        sub_D = D[np.ix_(indices, indices)]
        medoid_local_idx = np.argmin(sub_D.sum(axis=1))
        medoid_idx = indices[medoid_local_idx]
        total_cost += sum(D[i, medoid_idx] for i in indices)

    return total_cost