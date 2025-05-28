import json
import os

import pandas as pd
from openai import embeddings

from src import DATA_DIR, generate_embeddings, Node, Graph, GreedyCohesiveClustering


def test_iclr24_abstract_basic():
    input_path = os.path.join(DATA_DIR, 'unified_text', 'ICLR', f'ICLR_2024.json')
    output_path = os.path.join(DATA_DIR, 'emb', 'ICLR')
    model = 'text-embedding-3-large'
    inclusion = ['abstract', 'title', 'authors']

    embeddings = generate_embeddings(input_path, output_path, model, include=inclusion)

    nodes = []
    for key, value in embeddings.items():
        nodes.append(Node(key, value))

    graph = Graph(nodes, 32)
    print(GreedyCohesiveClustering(graph, 32))


if __name__ == '__main__':
    test_iclr24_abstract_basic()





