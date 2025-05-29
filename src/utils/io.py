import json
import numpy as np
import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_numpy(arr, path):
    np.save(path, arr)

def load_numpy(path):
    return np.load(path)
