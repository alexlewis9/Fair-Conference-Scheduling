import numpy as np


class Node:
    """
    Represents a generic node with an ID.
    """
    def __init__(self, node_id, emb):
        """
        Initializes the node with a unique ID.
        Args:
            node_id (str): Unique identifier for the node.
            emb (np.ndarray): Embedding (Coordinate) of the node.
        """
        self.id = node_id
        self.emb = np.array(emb)