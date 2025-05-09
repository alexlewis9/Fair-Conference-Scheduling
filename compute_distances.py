import json
from scipy.spatial.distance import cosine
import itertools

def compute_distances(embeddings_file, output_file):
    """
    Computes the cosine distances between embeddings stored in a JSON file.
    Saves the distances to a JSON file.
    """
    with open(embeddings_file, "r") as f:
        embeddings = json.load(f)
    
    distances = {}
    files = list(embeddings.keys())
    for file1, file2 in itertools.combinations(files, 2):
        emb1, emb2 = embeddings[file1], embeddings[file2]
        distance = cosine(emb1, emb2)
        distances[f"{file1} - {file2}"] = distance
    
    with open(output_file, "w") as f:
        json.dump(distances, f)

compute_distances("embeddings.json", "distances.json")