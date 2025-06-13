"""
K-means clustering with min and max cluster size.

https://github.com/joshlk/k-means-constrained
"""

import numpy as np
from k_means_constrained import KMeansConstrained

def kmeans_constrained(graph, k, upper_bound=None, lower_bound=None):
    model = KMeansConstrained(
        n_clusters=k,
        size_min=lower_bound,
        size_max=upper_bound,
    )
    labels = model.fit_predict(graph.embeddings)

    return labels