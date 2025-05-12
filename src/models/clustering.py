import math
import heapq
import numpy as np

from src.models.graph import Graph

def tau_closest_agents(agent_id, adj_matrix, tau) -> tuple[list[str], float]:
    """Return the list of tau-closest agents to agent.
    agent: agent's id
    agents: list of agents' id
    tau: threshold number (usually number of agents in a cluster)

    Return:
        - list of tau-closest agents' id
        - distance to the furthest agent
    """
    # Get distances from point i to all other points
    distances = adj_matrix[agent_id]
    # Get the distance to the tau-th closest agent
    dist_to_furthest_agent = np.partition(distances, tau - 1)[tau - 1]
    # Get the indices of the points within the ball
    tau_closest_agents = np.where(distances <= dist_to_furthest_agent)[0]
    return tau_closest_agents.tolist(), dist_to_furthest_agent


def SmallestAgentBall(adj_matrix, tau) -> list[str]:
    """Return the set of per_clusterclosest agents to the agent of the smallest ball.
    N: list of agents' id
    d: distance function
    tau: threshold number (usually number of agents in a cluster)
    """
    if len(adj_matrix) <= tau:
        return list(range(len(adj_matrix)))
    
    min_radius = float('inf')
    best_cluster = None

    for i in range(len(adj_matrix)):
        cluster_indices, dist_to_furthest_agent = tau_closest_agents(i, adj_matrix, tau)
        
        # Update if this ball is smaller
        if dist_to_furthest_agent < min_radius:
            min_radius = dist_to_furthest_agent
            best_cluster = cluster_indices

    return best_cluster


def GreedyCohesiveClustering(graph: Graph, k) -> list[list[str]]:
    """ Return the k cohesive clusters of agents by metric d. Each cluster is a list of id.
    agents: list of agents' id
    d: distance function
    k: number of clusters to return
    """
    n = len(graph.nodes)
    clusters = [] # each cluster is a list of id
    N = set(range(n))
    per_cluster = math.ceil(n/k)

    while N:
        # Create a submatrix for the remaining points
        remaining_indices_list = list(N)
        submatrix = graph.adj_matrix[np.ix_(remaining_indices_list, remaining_indices_list)]

        # Find the smallest ball in the remaining points
        C_j = SmallestAgentBall(submatrix, per_cluster)

        # Map cluster indices back to the original indices
        cluster_original_indices = [remaining_indices_list[i] for i in C_j]
        
        # Get the node IDs
        cluster_node_ids = [graph.nodes[i].id for i in cluster_original_indices]
        
        # Add the cluster to the result
        clusters.append(cluster_node_ids)
        
        # Remove the clustered points from the remaining set
        N -= set(cluster_original_indices)        

    # Add empty clusters if fewer than k clusters were created
    while len(clusters) < k:
        clusters.append([])

    return clusters