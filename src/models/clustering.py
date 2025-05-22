import math
import heapq
import numpy as np

from src.models.graph import Graph


def tau_closest_agents(agent_index, remaining_indices_list, adj_matrix, tau) -> tuple[list[int], float]:
    """Return the list of tau-closest agents to agent.
    agent: agent's id
    agents: list of agents' id
    tau: threshold number (usually number of agents in a cluster)

    Return:
        - list of tau-closest agents' id
        - distance to the furthest agent
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
    """Return the set of per_cluster-closest agents to the agent of the smallest ball.
    remaining_indices_list: list of remaining indices
    adj_matrix: adjacency matrix of the graph
    tau: threshold number (usually number of agents in a cluster)
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


def GreedyCohesiveClustering(graph: Graph, k) -> list[list[int]]:
    """ Return the k cohesive clusters of agents by metric d. Each cluster is a list of id.

    k: number of clusters to return
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