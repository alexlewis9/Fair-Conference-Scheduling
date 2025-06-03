import numpy as np
from sklearn.cluster import KMeans
# from pyclustering.cluster.kmedoids import kmedoids
import kmedoids

from src.eval.kmeans import kmeans_objective


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


