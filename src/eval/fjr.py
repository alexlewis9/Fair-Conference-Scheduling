import math
from src.eval.loss import avg_loss, max_loss
from src.models.clustering import SmallestAgentBall
from src.models.graph import Graph

def audit_fjr(graph: Graph, clustering: list[list[str]], loss):
    """ Return the approximated FJR score for the given clustering.
    Args:
        graph (Graph)
        clustering (list[list[str]]): A list of clusters, where each cluster is a list of node IDs.
        loss (str): The loss function to use. Choose from 'avg' or 'max'.
    """
    # Get a callable loss function
    if loss == 'avg': loss_func = avg_loss
    elif loss == 'max': loss_func = max_loss
    else: raise ValueError("Invalid loss function. Choose from 'avg' or 'max'.")


    n = len(graph.nodes)
    N = set(range(n)) # The global indices of agents from the Graph.

    theta = 0 # current FJR apx estimate
    k = graph.k # number of clusters
    per_cluster = math.ceil(n / k)
    clustering_index = [[graph.id_to_index[i] for i in row] for row in clustering] # Convert node IDs to global indices

    while len(N) >= n/k:
        # Find a new cohesive group
        remaining_indices_list = list(N)
        new_cluster = SmallestAgentBall(remaining_indices_list, graph.adj_matrix, per_cluster)

        # Update theta using the FJR violation due to S
        min_cluster_loss = float('inf') # current cluster, min_{i \in S} l_i(C(i))
        max_new_cluster_loss = float('-inf') # new cluster, max_{i \in S} l_i(S)
        min_cluster_loss_agent = None # agent with the smallest current loss, argmin_{i \in S} l_i(C(i))

        # i \in S
        for agent in new_cluster:
            # C(i)
            cluster = next((cluster for cluster in clustering_index if agent in cluster), None) # agent's current cluster
            # l_i(C(i))
            cluster_loss = loss_func(agent, cluster, graph.adj_matrix)
            # l_i(S)
            new_cluster_loss = loss_func(agent, new_cluster, graph.adj_matrix)

            if cluster_loss < min_cluster_loss:
                min_cluster_loss = cluster_loss
                # i*
                min_cluster_loss_agent = agent

            if new_cluster_loss > max_new_cluster_loss:
                max_new_cluster_loss = new_cluster_loss

        theta = max(theta, min_cluster_loss/max_new_cluster_loss)
        # Remove the agent with the smallest current loss
        N.discard(min_cluster_loss_agent)
    return theta
