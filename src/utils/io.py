import csv
import json
import numpy as np
import pandas as pd
import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(cfg, path):
    """Save a config dictionary to a YAML file."""
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def save_numpy(arr, path):
    np.save(path, arr)

def load_numpy(path):
    return np.load(path)


def save_csv(data, path):
    """
    Saves a list of dictionaries to a CSV file.

    Args:
        data (list[dict]): List of rows (each a dict of column-value pairs).
        path (str): Output file path.
    """
    if not data:
        raise ValueError("Data is empty. Nothing to save.")

    all_keys   = {k for row in data for k in row}

    preferred = "method" if "method" in all_keys else ("model" if "model" in all_keys else None)
    if preferred:
        fieldnames = [preferred] + sorted(all_keys - {preferred})
    else:
        fieldnames = sorted(all_keys)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

