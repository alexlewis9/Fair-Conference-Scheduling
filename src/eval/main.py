import numpy as np
from src.eval.intra_cluster_pairwise import intra_cluster_pairwise_cost
from src.eval.kmeans import kmeans_objective
from src.eval.kmedoids import kmedoid_objective
from src.eval.core import run_core
from src.eval.FJR import run_FJR

def evaluate_cluster(graph, clustering, metric=None):
    n = len(graph.nodes)
    k = len(clustering)
    d = graph.adj_mat
    M = [None] * n
    for cluster_idx, cluster in enumerate(clustering):
        for node_id in cluster:
            M[graph.id_to_index[node_id]] = cluster_idx
    # theta = 1
    theta = np.max(graph.adj_mat)

    return {
        "k-means": kmeans_objective(graph, clustering),
        "k-medoids": kmedoid_objective(graph, clustering, metric),
        "avg_compactness": intra_cluster_pairwise_cost(graph, clustering, inner='avg', outer='avg', metric=metric),
        "avg_diameter": intra_cluster_pairwise_cost(graph, clustering, inner='max', outer='avg', metric=metric),
        "worst_avg_spread": intra_cluster_pairwise_cost(graph, clustering, inner='avg', outer='max', metric=metric),
        "max_diameter": intra_cluster_pairwise_cost(graph, clustering, inner='max', outer='max', metric=metric),
        "alpha-core": run_core(n, k, d, M, theta),
        "alpha-fjr": run_FJR(n, k, d, M, theta)
    }