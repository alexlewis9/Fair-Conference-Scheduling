"""
K-means clustering with min and max cluster size.

https://github.com/joshlk/k-means-constrained
"""

import math
import numpy as np
from k_means_constrained import KMeansConstrained

def kmeans_constrained(graph, k, random_state=None):
    n = len(graph.embeddings)
    floor = n // k
    ceil = math.ceil(n / k)
    min_size = max(1, floor - 1)
    max_size = ceil + 1

    model = KMeansConstrained(
        n_clusters=k,
        size_min=min_size,
        size_max=max_size,
        random_state=random_state
    )
    labels = model.fit_predict(graph.embeddings)

    return labels