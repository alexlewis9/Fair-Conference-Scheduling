import numpy as np

from src.models.graph import Graph


def kmeans_objective(graph: Graph, clusters):
    total_wcss = 0.0

    for cluster in clusters:
        if cluster == []:
            continue
        lst = []
        for node_id in cluster:
            node = graph.get_node(node_id)
            lst.append(node.emb)
        embeddings = np.array(lst)

        # Get all embeddings in this cluster
        # embeddings = np.array([graph.get_node(node_id).emb for node_id in cluster])

        # Compute centroid
        centroid = embeddings.mean(axis=0)

        # Sum of squared distances to centroid
        sq_dists = ((embeddings - centroid) ** 2).sum(axis=1)
        total_wcss += sq_dists.sum()

    return total_wcss