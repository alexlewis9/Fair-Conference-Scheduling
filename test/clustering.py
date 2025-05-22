from src import Node, Graph
from src import GreedyCohesiveClustering
from src.eval.fjr import audit_fjr


def test_clustering():
    nodes = [
        Node('0',[0, 0]),
             Node ('1', [2, 0]),
             Node ('2', [0, 1]),
             Node ('3', [2, 1])
             ]
    graph = Graph(nodes, 2)
    clustering = GreedyCohesiveClustering(graph, graph.k)
    print(clustering)
    clustering = [['0', '1'], ['2', '3']]
    print(audit_fjr(graph, clustering, 'avg'))


if __name__ == '__main__':
    test_clustering()