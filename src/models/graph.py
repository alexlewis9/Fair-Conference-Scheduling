import numpy as np
from scipy.spatial.distance import cdist

from src.models.node import Node


class Graph:
    """
    Represents a graph with nodes, embeddings, adjacency matrix, and clustering.
    """
    def __init__(self, nodes, k, d='euclidean', original_clusters=None):
        """
        Initializes the graph.

        Args:
            nodes (list or dict): A list or dictionary of node objects.
            k (int): Number of clusters.
            d (str): Distance metric (e.g. 'cityblock', 'euclidean', 'cosine').
            original_clusters (list, optional): Original clusters if provided.
        Attributes:
            embeddings (np.ndarray): Embedding matrix where row i corresponds to the embedding of nodes[i].
        """
        if isinstance(nodes, dict):
            self.nodes = [Node(nid, emb) for nid, emb in nodes.items()]
        elif isinstance(nodes, list):
            self.nodes = nodes[:]
        self.id_to_index = {node.id: i for i, node in enumerate(self.nodes)}
        self.embeddings = self._process_emb(self.nodes)
        self.k = k
        self.d = d
        self.original_clusters = original_clusters if original_clusters else []
        self.adj_mat = self._compute_adj_matrix()
        
    def _process_emb(self, nodes):
        """
        Validates that all node embeddings have the same shape and creates embedding matrix.
        
        Args:
            nodes (list): List of node objects.
            
        Returns:
            np.ndarray: Matrix of node embeddings.
            
        Raises:
            ValueError: If node embeddings have different shapes.
        """
        if not nodes:
            return np.array([])

        first_shape = nodes[0].emb.shape
        for node in nodes[1:]:
            if node.emb.shape != first_shape:
                raise ValueError(
                    f"Node embeddings must have same shape. Found shapes {first_shape} and {node.emb.shape}")

        return np.array([node.emb for node in nodes])

    def _compute_adj_matrix(self):
        """
        Computes the adjacency matrix using the specified distance metric.

        Returns:
            np.ndarray: Adjacency matrix storing pairwise distances.
        """
        return cdist(self.embeddings, self.embeddings, metric=self.d)

    def get_node(self, node_id):
        """
        Retrieves a node by its ID.

        Args:
            node_id (int): The ID of the node to retrieve.

        Returns:
            Node: The node with the specified ID, or None if not found.
        """
        index = self.id_to_index[node_id]
        return self.nodes[index]

