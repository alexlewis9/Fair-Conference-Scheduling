import math
import heapq
import numpy as np

from src.models.graph import Graph


def tau_closest_agents(agent_index, remaining_indices_list, adj_matrix, tau) -> tuple[list[int], float]:
    """
    Find the tau closest agents to a given agent based on the provided adjacency matrix.

    Args:
        agent_index (int): Index of the agent (row in adj_matrix) to find neighbors for.
        remaining_indices_list (list[int]): List of candidate agent indices (global indices) to search within.
        adj_matrix (np.ndarray): A 2D numpy array representing pairwise distances between agents.
        tau (int): Number of closest agents to select (including the agent itself if present).

    Returns:
        tuple[list[int], float]:
            - List of the global indices of the tau closest agents.
            - Distance to the furthest agent in this tau-sized neighborhood.
    """
    # Get distances from point i to all other points
    distances = adj_matrix[agent_index][remaining_indices_list]
    # Use a heap to get the tau closest agents
    tau_closest = heapq.nsmallest(tau, zip(remaining_indices_list, distances), key=lambda x: x[1])
    # Extract the indices of the closest agents
    cluster_indices = [i for i, _ in tau_closest]
    # Get the distance to the furthest agent in the tau closest agents
    dist_to_furthest_agent = tau_closest[-1][1]

    return cluster_indices, dist_to_furthest_agent


def SmallestAgentBall(remaining_indices_list, adj_matrix, tau) -> list[int]:
    """
    Identify the tightest group ("smallest ball") of tau agents from the remaining set,
    minimizing the radius (distance to the furthest member from a center).

    Args:
        remaining_indices_list (list[int]): Global indices of unclustered agents.
        adj_matrix (np.ndarray): A 2D numpy array representing pairwise distances between agents.
        tau (int): Desired number of agents in the group.

    Returns:
        list[int]: Global indices of the tau agents forming the smallest-radius ball.
                   If the number of remaining agents is less than or equal to tau, returns all.
    """
    if len(remaining_indices_list) <= tau:
        return remaining_indices_list
    
    min_radius = float('inf')
    best_cluster = None

    for i in remaining_indices_list:
        cluster_indices, dist_to_furthest_agent = tau_closest_agents(i, remaining_indices_list, adj_matrix, tau)
        
        # Update if this ball is smaller
        if dist_to_furthest_agent < min_radius:
            min_radius = dist_to_furthest_agent
            best_cluster = cluster_indices

    return best_cluster


def GreedyCohesiveClustering(graph: Graph, k) -> list[list[str]]:
    """
    Partition agents into k cohesive clusters using a greedy approach that minimizes
    the radius of each cluster.

    Each cluster contains approximately n/k agents (rounded up), and is selected
    based on the agent whose tau-neighborhood forms the smallest "ball" in terms of max distance.

    Args:
        graph (Graph): Graph object containing:
            - nodes: a list of node objects, each with an `id` attribute.
            - adj_matrix: a 2D numpy array representing pairwise distances between nodes.
        k (int): Number of clusters to produce.

    Returns:
        list[list[str]]: A list of k clusters, where each cluster is a list of node IDs.
                         If fewer than k meaningful clusters are found, remaining entries are empty lists.
    """
    n = len(graph.nodes)
    clusters = [] # each cluster is a list of id
    N = set(range(n))
    per_cluster = math.ceil(n/k)

    while len(N) >= per_cluster:
        # Create a submatrix for the remaining points
        remaining_indices_list = list(N)

        # Find the smallest ball in the remaining points
        C_j = SmallestAgentBall(remaining_indices_list, graph.adj_matrix, per_cluster)

        # Get the node IDs
        cluster_node_ids = [graph.nodes[i].id for i in C_j]
        
        # Add the cluster to the result
        clusters.append(cluster_node_ids)
        
        # Remove the clustered points from the remaining set
        N -= set(C_j)

    if N:
        remaining_node_ids = [graph.nodes[i].id for i in N]
        clusters.append(remaining_node_ids)

    # Add empty clusters if fewer than k clusters were created
    while len(clusters) < k:
        clusters.append([])

    return clusters