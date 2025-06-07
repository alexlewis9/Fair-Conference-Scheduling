from src.eval.metrics.FJR import get_FJR
from src.eval.metrics.appr_FJR import get_appr_FJR
from src.eval.metrics.core import get_core
from src.eval.metrics.intra_cluster_pairwise import intra_cluster_pairwise_cost
from src.eval.metrics.kmeans import kmeans_objective
from src.eval.metrics.kmedoids import kmedoid_objective
from src.eval.metrics.silhoutte import get_silhouette


def evaluate_cluster(graph, clustering, metric=None, loss='avg'):
    theta = get_appr_FJR(graph, clustering, loss)
    core = get_core(graph, clustering, theta, loss)
    fjr = get_FJR(graph, clustering, theta, loss)
    return {
        'core': core,
        'fjr': fjr,
        'silhouette': get_silhouette(graph, clustering, metric=metric),
        "k-means": kmeans_objective(graph, clustering),
        "k-medoids": kmedoid_objective(graph, clustering, metric),
        "avg_compactness": intra_cluster_pairwise_cost(graph, clustering, inner='avg', outer='avg', metric=metric),
        "avg_diameter": intra_cluster_pairwise_cost(graph, clustering, inner='max', outer='avg', metric=metric),
        "worst_avg_spread": intra_cluster_pairwise_cost(graph, clustering, inner='avg', outer='max', metric=metric),
        "max_diameter": intra_cluster_pairwise_cost(graph, clustering, inner='max', outer='max', metric=metric)
    }