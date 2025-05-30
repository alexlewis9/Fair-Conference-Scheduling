import numpy as np
from sklearn.cluster import KMeans
# from pyclustering.cluster.kmedoids import kmedoids
import kmedoids


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
    clustering = process.fit_predict(graph.embeddings) # List of cluster labels of each node
    return format_clustering(graph, clustering)

# def kmedoids_clustering(graph, k, seed=42):
#     np.random.seed(seed)
#     # Requires indices of initial medoids
#     initial_medoids = np.random.choice(len(graph.nodes), size=k, replace=False)
#     kmedoids_instance = kmedoids(data=graph.embeddings, initial_index_medoids=initial_medoids)
#     kmedoids_instance.process()
#     clusters = kmedoids_instance.get_clusters()
#     formatted = []
#     for cluster in clusters:
#         formatted.append(graph.nodes[cluster].id)
#     return formatted

def kmedoids_clustering(graph, k):
    adj_mat = graph.adj_mat
    # Perform clustering
    result = kmedoids.fasterpam(adj_mat, medoids=k)
    return format_clustering(graph, result.labels)


