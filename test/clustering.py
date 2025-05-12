from src import Node, Graph
from src import GreedyCohesiveClustering


def test_clustering():
    nodes = [Node(i) for i in range(4)]
    emb = [[0, 0], [2, 0], [0, 1], [2, 1]]
    graph = Graph(nodes, emb, 2)
    print(GreedyCohesiveClustering(graph, graph.k))


if __name__ == '__main__':
    test_clustering()