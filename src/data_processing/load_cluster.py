# TODO: save and load clusters
import json


def load_cluster_from_data(path, to_dict=False):
    with open(path, "r") as f:
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
        clustering = []
        labels = []
        for k, v in clustering_og.items():
            clustering.append(v)
            labels.append(k)
        return clustering, labels
