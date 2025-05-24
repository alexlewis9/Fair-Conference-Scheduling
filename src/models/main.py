import json
import os.path
from datetime import datetime

from src.models.node import Node
from src.models.graph import Graph
from src.models.clustering import GreedyCohesiveClustering

def get_fair_cluster(input_json, output_path, num_clusters, metric='euclidean'):
    """ Return the fair cluster for the given embedding JSON as JSON in the output path.

    """
    with open(input_json, 'r') as f:
        input_data = json.load(f)

    embeddings = input_data['emb']
    nodes = [Node(paper_id, emb) for paper_id, emb in embeddings.items()] #TODO: refactor into Node
    graph = Graph(nodes, num_clusters, metric)

    clusters = GreedyCohesiveClustering(graph, num_clusters)

    output_data = {
        'emb_config': input_data['config'],
        'cluster_config': {
            'num_clusters': num_clusters,
            'metric': metric
        },
        'fair_clusters': clusters
    }
    # emb_config_input_path = json.dumps(input_data['config']['input_file'])
    emb_config_input_path = input_data['config']['input_file']
    filename = os.path.splitext(os.path.basename(emb_config_input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(output_path, filename)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

