import numpy as np
from scipy.spatial.distance import cdist

class Graph:
    """
    Represents a graph with nodes, embeddings, adjacency matrix, and clustering.
    """
    def __init__(self, nodes, k, d='euclidean', original_clusters=None):
        """
        Initializes the graph.

        Args:
            nodes (list): A list of node objects.
            nodes (dict): A dictionary mapping node IDs to node objects.
            embeddings (np.ndarray): Embedding matrix where each row corresponds to a node's embedding.
            k (int): Number of clusters.
            d (str): Distance metric (e.g. 'cityblock', 'euclidean', 'cosine').
            original_clusters (list, optional): Original clusters if provided.
        """
        self.nodes = nodes
        self.nodes_dict = {node.id: node for node in nodes}
        self.embeddings = np.array([node.emb for node in nodes])
        self.k = k
        self.d = d
        self.original_clusters = original_clusters if original_clusters else []
        self.adj_matrix = self._compute_adj_matrix()

    def _compute_adj_matrix(self):
        """
        Computes the adjacency matrix using the specified distance metric.

        Returns:
            np.ndarray: Adjacency matrix storing pairwise distances.
        """
        return cdist(self.embeddings, self.embeddings, metric=self.d)

    def get_node_by_id(self, node_id):
        """
        Retrieves a node by its ID.

        Args:
            node_id (int): The ID of the node to retrieve.

        Returns:
            Node: The node with the specified ID, or None if not found.
        """
        return self.nodes.get(node_id, None)