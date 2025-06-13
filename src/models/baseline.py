import numpy as np
from sklearn.cluster import KMeans
# from pyclustering.cluster.kmedoids import kmedoids
import kmedoids
from src.models.same_size_kmeans_elki import same_size_kmeans_elki
from src.models.kmeans_constrained import kmeans_constrained
from src.eval.metrics.kmeans import kmeans_objective
from src.models.same_size_kmeans_linear_sum_assignment import same_size_kmeans_linear_sum_assignment
from src.models.same_size_kmeans_greedy import same_size_kmeans_greedy
import math


def format_clustering(graph, clustering):
    formatted = {}
    for i in range(len(clustering)):
        label = int(clustering[i])
        if not formatted.get(label):
            formatted[label] = [graph.nodes[i].id]
        else:
            formatted[label].append(graph.nodes[i].id)
    return [val for _, val in formatted.items()]

def kmeans_clustering(graph, k):
    process = KMeans(n_clusters=k)
    best_trial = None
    best_score = np.inf
    for _ in range(20):
        clustering = process.fit_predict(graph.embeddings) # List of cluster labels of each node
        clustering = format_clustering(graph, clustering)
        score = kmeans_objective(graph, clustering)
        if score < best_score:
            best_score = score
            best_trial = clustering
    return best_trial

def kmedoids_clustering(graph, k):
    adj_mat = graph.adj_mat
    # Perform clustering
    best_trial = None
    best_score = np.inf
    for _ in range(20):
        clustering = kmedoids.fasterpam(adj_mat, medoids=k).labels
        clustering = format_clustering(graph, clustering)
        score = kmeans_objective(graph, clustering)
        if score < best_score:
            best_score = score
            best_trial = clustering
    return best_trial

def same_size_kmeans_elki_clustering(graph, k):
    _, assignments = same_size_kmeans_elki(graph, k)
    return format_clustering(graph, assignments)

def kmeans_constrained_clustering(graph, k):
    n = len(graph.embeddings)
    floor = n // k
    ceil = math.ceil(n / k)
    min_size = max(1, floor - 1)
    max_size = ceil + 1

    clustering = kmeans_constrained(graph, k, upper_bound=max_size, lower_bound=min_size)
    return format_clustering(graph, clustering)

def same_size_kmeans_linear_sum_assignment_clustering(graph, k):
    clustering = same_size_kmeans_linear_sum_assignment(graph, k)
    return format_clustering(graph, clustering)

def same_size_kmeans_greedy_clustering(graph, k):
    clustering = same_size_kmeans_greedy(graph, k)
    return format_clustering(graph, clustering)

def kmeans_constrained_nolowerbound_clustering(graph, k):
    clustering = kmeans_constrained(graph, k)
    return format_clustering(graph, clustering)