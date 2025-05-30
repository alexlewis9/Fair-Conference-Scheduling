import argparse

from src.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    model = cfg["model"]
    if model == 'GreedyCohesive':
        print("Greedy Cohesive Clustering")
    elif model == 'KMeans':
        print("KMeans Clustering")
    elif model == 'KMedoids':
        print("KMedoids Clustering")

