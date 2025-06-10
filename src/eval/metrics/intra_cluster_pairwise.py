import numpy as np
from scipy.spatial.distance import cdist

# TODO: test this
def intra_cluster_pairwise_cost(graph, clustering,
                                 inner='max', outer='max', metric=None):
    """
    Computes clustering cost as aggregation (outer) over each cluster of
    aggregation (inner) of pairwise distances within the cluster.

    Args:
        graph (Graph): Graph object containing embeddings and id_to_index.
        clustering (list[list[int]]): List of clusters, each a list of node IDs.
        inner (str): 'max' or 'avg' for intra-cluster aggregation.
        outer (str): 'max' or 'avg' for inter-cluster aggregation.
        metric (str): Optional distance metric (default: graph.d).

    Returns:
        float: Total cost.
    """
    assert inner in ('max', 'avg'), "inner must be 'max' or 'avg'"
    assert outer in ('max', 'avg'), "outer must be 'max' or 'avg'"

    metric = metric if metric else graph.d
    D = graph.adj_mat if metric == graph.d else cdist(graph.embeddings, graph.embeddings, metric=metric)

    cluster_costs = []

    for cluster in clustering:
        indices = [graph.id_to_index[nid] for nid in cluster]
        if len(indices) <= 1:
            cluster_costs.append(0.0)
            continue
        sub_D = D[np.ix_(indices, indices)]
        triu = sub_D[np.triu_indices_from(sub_D, k=1)]
        if inner == 'max':
            cluster_costs.append(np.max(triu))
        else:
            cluster_costs.append(np.mean(triu))

    return max(cluster_costs) if outer == 'max' else np.mean(cluster_costs)
