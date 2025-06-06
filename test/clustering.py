from src import Node, Graph
from src import greedy_cohesive_clustering
from src.models.baseline import kmeans_clustering, kmedoids_clustering


def test_clustering():
    nodes = [
        Node('0',[0, 0]),
             Node ('1', [2, 0]),
             Node ('2', [0, 1]),
             Node ('3', [2, 1])
             ]
    graph = Graph(nodes, 2)
    clustering = greedy_cohesive_clustering(graph, graph.k)
    print(clustering)
    result = [['0', '1'], ['2', '3']]
    kmeans = kmeans_clustering(graph, graph.k)
    print(kmeans)
    kmedoids = kmedoids_clustering(graph, graph.k)
    print(kmedoids)


if __name__ == '__main__':
    test_clustering()