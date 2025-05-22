import numpy as np


def avg_loss(agent_index: int, cluster: list, adj_matrix: np.ndarray) -> float:
    """Return the average loss for the given agent and cluster."""
    if not cluster:
        return 0
    return sum(adj_matrix[agent_index][i] for i in cluster) / len(cluster)


def max_loss(agent_index: int, cluster: list, adj_matrix: np.ndarray) -> float:
    """Return the maximum loss for the given agent and cluster."""
    if not cluster:
        return 0
    return max(adj_matrix[agent_index][i] for i in cluster)
