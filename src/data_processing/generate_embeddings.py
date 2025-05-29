import os
import json
from datetime import datetime

from src.data_processing.encoder import Encoder

import os
import json


def generate_embeddings(input_path, output_path, model_name, include=None, exclude=None, stride=0):
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
    """
    if include is None:
        include = []
    if exclude is None:
        exclude = []


    model = Encoder(model_name, stride=stride)
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(output_path, input_filename)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{timestamp}.json")

    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize with run config
    embeddings = {
                    'config': {
                            'model_name': model_name,
                            'include': include,
                            'exclude': exclude,
                            'input_file': input_path,
                            'output_file': output_file,
                            'stride': model.stride,
                            'max_tokens': model.max_tokens
                            },
                    'emb': {},
                    'raw_emb': {} # i.e. pre-reconstructed emb
                  }

    for entry in data:
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
        print(entry_str)

        # Encode and store
        embeddings['emb'][entry['id']], embeddings['raw_emb'][entry['id']] = model.encode_pre_recon(entry_str)


    # Save to output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    return embeddings



