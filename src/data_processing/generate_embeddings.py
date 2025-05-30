import os
import json
from datetime import datetime

from src.data_processing.encoder import Encoder

import os
import json


def generate_embeddings(input_path, output_path, model_name,
                        include=None,
                        exclude=None,
                        stride=0,
                        max_tokens=0,
                        verbose = True):
    """
    Generates embeddings for entries in a JSON file and saves the result to an output JSON.

    For each entry (dictionary) in the input JSON, the function generates embeddings for its values
    based on the specified `include` and `exclude` lists:
      - An id must be included and excepted from any rules.
      - If both `include` and `exclude` are empty or None, all keys are processed.
      - If only `include` is provided, only keys in `include` will be embedded.
      - If only `exclude` is provided, all keys except those in `exclude` will be embedded.
      - If both are provided, `exclude` is applied after `include`.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output directory.
        model_name (str): Name of the model used to generate embeddings.
        include (list, optional): Keys to include in embedding generation.
        exclude (list, optional): Keys to exclude from embedding generation.

    Returns:
        embeddings (dict): Embeddings generated for each entry in the input JSON and config.
        raw_emb (dict): Raw chunks embeddings that are not post-processed (i.e., not normalized).
    """
    if include is None:
        include = []
    if exclude is None:
        exclude = []


    model = Encoder(model_name, stride=stride, max_tokens=max_tokens)

    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize with run config
    embeddings = {}
    raw_emb = {}

    # Prepare log
    log_lines = []
    start_time = datetime.now()
    log_lines.append(f"Embedding started at {start_time.isoformat()}")
    log_lines.append(f"Model: {model_name}")
    log_lines.append(f"Input file: {input_path}")

    for entry in data:
        entry_id = str(entry.get("id", ""))
        try:
            # Determine which keys to include
            if include:
                keys_to_use = set(include)
            else:
                keys_to_use = set(entry.keys())

            if exclude:
                keys_to_use -= set(exclude)

            # Always include 'id' if present in the entry
            if 'id' in entry:
                keys_to_use.add('id')
            else:
                raise ValueError("Each entry in the input JSON must have an 'id' field.")

            # Construct the string to embed
            filtered_entry = {k: entry[k] for k in keys_to_use if k in entry}
            entry_str = json.dumps(filtered_entry, ensure_ascii=False)

            # Encode and store
            embeddings[entry['id']], raw_emb[entry['id']] = model.encode_pre_recon(entry_str)
            if verbose:
                print(f"[OK] {entry_id}")
            log_lines.append(f"[OK] {entry_id}")

        except Exception as e:
            err_msg = f"[ERROR] {entry_id}: {str(e)}"
            log_lines.append(err_msg)
            if verbose:
                print(err_msg)

    return embeddings, raw_emb, log_lines



