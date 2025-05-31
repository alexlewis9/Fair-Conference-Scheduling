import os

from src.models.main import get_fair_cluster
from src import DATA_DIR


input_path = os.path.join(DATA_DIR, "emb", "test", "20250521_1825.json")
output_path = os.path.join(DATA_DIR, "fair_cluster")
get_fair_cluster(input_path, output_path, 2)