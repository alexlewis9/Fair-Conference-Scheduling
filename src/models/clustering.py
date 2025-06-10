import math
import heapq
import numpy as np

from src.models.graph import Graph


def GreedyCohesiveClustering(graph: Graph, k) -> list[list[int]]:
    """ Return the k cohesive clusters of agents by metric d. Each cluster is a list of id.
    agents: list of agents' id
    d: distance function
    k: number of clusters to return
    """
    n = len(graph.nodes)
    clusters = [] # each cluster is a list of id
    remaining = set(range(n))
    per_cluster = math.ceil(n/k)
    removed = set()

    # Create list of neighbors sorted by distance for each node
    sorted_neighbors = []
    for i in range(n):
        dists = [(j, graph.adj_matrix[i, j]) for j in range(n) if j != i]
        dists.sort(key=lambda x: x[1])
        sorted_neighbors.append([j for j, _ in dists])

    # Initialize closest neighbors for each node
    closest_neighbors = [set(sorted_neighbors[i][:per_cluster]) for i in range(n)]

    # Initialize pointers to track position in sorted neighbors list for each node
    pointers = [per_cluster - 1 for _ in range(n)]

    while len(remaining) >= per_cluster:
        min_radius = float('inf')
        best_cluster = None

        for i in remaining:
            # Remove any removed nodes from closest_neighbors[i]
            closest_neighbors[i] = closest_neighbors[i].difference(removed)
            l = per_cluster - len(closest_neighbors[i])
            ptr = pointers[i]
            # Add neighbors until we have per_cluster valid nodes in closest neighbors
            while l > 0:
                ptr += 1
                neighbor = sorted_neighbors[i][ptr]
                if neighbor not in removed:
                    l -= 1
                    closest_neighbors[i].add(neighbor)
            pointers[i] = ptr

            farthest_node = sorted_neighbors[i][pointers[i]]
            radius = graph.adj_matrix[i, farthest_node]
            if radius < min_radius:
                min_radius = radius
                best_cluster = closest_neighbors[i]

        clusters.append([graph.nodes[i].id for i in best_cluster])
        removed.update(best_cluster)
        remaining.difference_update(best_cluster)

    if remaining:
        clusters.append([graph.nodes[i].id for i in remaining])

    # Add empty clusters if fewer than k clusters were created
    while len(clusters) < k:
        clusters.append([])

    return clusters