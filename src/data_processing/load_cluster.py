# TODO: save and load clusters
import json


def load_cluster_from_data(path, to_dict=False):
    with open(path, "r", encoding='utf-8') as f:
        file = json.load(f)
    clustering_og = {}
    for entry in file:
        if entry['session'] not in clustering_og:
            clustering_og[entry['session']] = [entry['id']]
        else:
            clustering_og[entry['session']].append(entry['id'])

    if to_dict:
        return clustering_og
    else:
        clusters = []
        labels = []
        for key, value in clustering_og.items():
            labels.append(key)
            clusters.append(value)
        return clusters, labels