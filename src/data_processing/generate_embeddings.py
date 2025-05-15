import os
import json
from src.data_processing.encoder import Encoder

def generate_embeddings(input_path, output_path, model_name: str):
    """
    Return embeddings in a JSON by processing the input JSON file to generate embeddings for each complete entry.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        model_name
    """
    model = Encoder(model_name)
    input_filename = os.path.splitext(os.path.basename(input_path))[0]  # For the directory name
    os.makedirs(os.path.join(output_path, input_filename), exist_ok=True)  # make sure the directory exists

    output_name = f"{model_name}.json"
    output_path = os.path.join(output_path, input_filename, output_name)

    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Generate embeddings for each complete entry
    embeddings = {}
    for entry in data:
        entry_str = json.dumps(entry)  # Convert the entire entry to string
        embeddings[entry['id']] = model.encode(entry_str)

    # Save embeddings to an output file
    with open(output_path, 'w') as f:
        json.dump(embeddings, f)



