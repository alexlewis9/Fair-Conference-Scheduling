from src import Node, Graph
from src import GreedyCohesiveClustering


def test_clustering():
    nodes = [
        Node(0,[0, 0]),
             Node (1, [2, 0]),
             Node (2, [0, 1]),
             Node (3, [2, 1])
             ]
    graph = Graph(nodes, 2)
    print(GreedyCohesiveClustering(graph, graph.k))


if __name__ == '__main__':
    test_clustering()