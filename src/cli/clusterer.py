import argparse
import os
import warnings
from datetime import datetime
import numpy as np
from scipy.optimize import linear_sum_assignment

from src import greedy_cohesive_clustering
from src.data_processing.load_cluster import load_cluster_from_data
from src.eval.main import evaluate_cluster
from src.models.baseline import kmeans_clustering, kmedoids_clustering
from src.models.graph import Graph
from src.utils.io import load_yaml, load_json, save_csv, save_yaml

from pathlib import Path
import warnings

# ---------- helpers ---------------------------------------------------------
def jaccard_similarity(set1, set2):
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union != 0 else 0

def align_clusterings(reference_clusters: list[list[str]],
                      target_clusters : list[list[str]]) -> list[list[str]]:
    """
    Reorders `target_clusters` so they best match `reference_clusters`
    (min-cost assignment on 1-Jaccard). Returns the reordered clusters.
    """
    ref_sets    = [set(c) for c in reference_clusters]
    tgt_sets    = [set(c) for c in target_clusters]
    n           = len(ref_sets)
    cost_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = 1 - jaccard_similarity(ref_sets[i], tgt_sets[j])

    _, col_ind = linear_sum_assignment(cost_matrix)
    return [target_clusters[j] for j in col_ind]

def build_cluster_rows(clusterings: dict[str, list[list[str]]],
                       column_labels: list[str]) -> list[dict]:
    """
    Returns a list of {'method': …, <session_0>: 'a,b,c', …} dictionaries
    ready for `save_csv`.
    """
    rows = []
    k    = len(column_labels)           # number of columns to emit

    for method, clusters in clusterings.items():
        row = {'method': method}
        for i, label in enumerate(column_labels):
            row[label] = ",".join(clusters[i]) if i < len(clusters) else ""
        rows.append(row)

    return rows

def load_embeddings(embed_path):
    """Return (embeddings, emb_cfg) loaded from <embed_path>."""
    emb_cfg = load_yaml(os.path.join(embed_path, "config.yaml"))
    emb     = load_json(os.path.join(embed_path, "emb.json"))
    return emb, emb_cfg


def build_graph(emb, k: int, metric: str):
    """Create Graph, defaulting k to len(baseline_cluster) if k == 0."""
    if k <= 0:
        raise ValueError("k must be >0 at graph-creation time")
    return Graph(emb, k, d=metric)


def get_baseline_clusters(data_path, to_dict=False):
    """Load the baseline clustering once and memoise it."""
    return load_cluster_from_data(data_path, to_dict=to_dict)


def ensure_baseline(models: list[str], k: int):
    """Warn the user if k==0 and baseline isn't requested."""
    if k == 0 and "Baseline" not in models:
        warnings.warn(
            "k==0 → using baseline cluster count, "
            "but 'Baseline' not in models list; adding it automatically.",
            UserWarning,
        )
        models.append("Baseline")


def evaluate_models(graph, models, clusterings, metric):
    """Return list[dict] with evaluation rows."""
    results = []
    for model in models:
        clustering = clusterings[model]
        if isinstance(clustering, dict):
            clustering = list(clustering.values())
            clustering[model] = clustering
        row        = {"model": model, **evaluate_cluster(graph, clustering, metric)}
        results.append(row)
    return results


# ---------- main ------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg         = load_yaml(args.config)
    models      = cfg["models"][:]          # copy to avoid side-effects
    embed_path  = cfg["embed_path"]
    output_path = cfg["output_path"]
    data_path   = cfg["data_path"]
    metric      = cfg["metric"]
    k           = cfg["k"]

    # 1) embeddings & graph ---------------------------------------------------
    emb, emb_cfg      = load_embeddings(embed_path)

    ensure_baseline(models, k)

    # baseline clusterings may be needed for k==0 *and/or* evaluation
    baseline_clusters, labels = get_baseline_clusters(data_path) if "Baseline" in models else None

    # choose k
    effective_k = len(baseline_clusters) if k == 0 else k
    graph       = build_graph(emb, effective_k, metric)

    # 2) collect clusterings --------------------------------------------------
    clusterings = {}

    if "Baseline" in models:
        clusterings["Baseline"] = baseline_clusters

    if "GreedyCohesive" in models:
        clusterings["GreedyCohesive"] = greedy_cohesive_clustering(graph, effective_k)

    if "KMedoids" in models:
        clusterings["KMedoids"] = kmedoids_clustering(graph, effective_k)

    if "KMeans" in models:
        clusterings["KMeans"] = kmeans_clustering(graph, effective_k)

    # ...add more models here...

    # 3) evaluate -------------------------------------------------------------
    evaluations = evaluate_models(graph, models, clusterings, metric)

    # ------------------------------------------------------------------
    # Align ALL methods to a single reference, decide the column labels
    # ------------------------------------------------------------------
    if "Baseline" in clusterings:
        ref_clusters = clusterings["Baseline"]
        column_labels = labels[:]
    else:
        # pick the first model as reference and create synthetic labels
        first_model = next(iter(clusterings))
        ref_clusters = clusterings[first_model]
        column_labels = [f"cluster_{i}" for i in range(len(ref_clusters))]

    # Re-order every method’s clusters so they correspond to ref_clusters
    for m, clust in clusterings.items():
        if clust is ref_clusters:
            continue
        clusterings[m] = align_clusterings(ref_clusters, clust)

    # 4) save ------------------------------------------------------
    output_path = os.path.join(output_path, timestamp)
    os.makedirs(output_path, exist_ok=True)

    output_eval = os.path.join(output_path, "eval.csv")
    save_csv(evaluations, output_eval)

    cluster_rows = build_cluster_rows(clusterings, column_labels)
    output_clustering = os.path.join(output_path, "clusters.csv")
    save_csv(cluster_rows, output_clustering)

    cfg["effective_k"] = effective_k
    output_clusterer_cfg = os.path.join(output_path, "clusterer.cfg")
    save_yaml(cfg, output_clusterer_cfg)

    emb_cfg_path = os.path.join(output_path, "emb.yaml")
    save_yaml(emb_cfg, emb_cfg_path)

if __name__ == "__main__":
    main()










