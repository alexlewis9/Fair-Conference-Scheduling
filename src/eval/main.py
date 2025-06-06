from src.eval.intra_cluster_pairwise import intra_cluster_pairwise_cost
from src.eval.kmeans import kmeans_objective
from src.eval.kmedoids import kmedoid_objective


def evaluate_cluster(graph, clustering, metric=None):
    return {
        "k-means": kmeans_objective(graph, clustering),
        "k-medoids": kmedoid_objective(graph, clustering, metric),
        "avg_compactness": intra_cluster_pairwise_cost(graph, clustering, inner='avg', outer='avg', metric=metric),
        "avg_diameter": intra_cluster_pairwise_cost(graph, clustering, inner='max', outer='avg', metric=metric),
        "worst_avg_spread": intra_cluster_pairwise_cost(graph, clustering, inner='avg', outer='max', metric=metric),
        "max_diameter": intra_cluster_pairwise_cost(graph, clustering, inner='max', outer='max', metric=metric)
    }