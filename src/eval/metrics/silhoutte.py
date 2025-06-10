from sklearn.metrics import silhouette_score

def get_silhouette(graph, clustering, metric ='euclidean'):
    flatten = graph.flatten_clusters(clustering)
    return silhouette_score(graph.adj_mat, flatten, metric=metric)