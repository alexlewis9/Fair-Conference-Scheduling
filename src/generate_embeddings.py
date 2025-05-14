import os
import json
import argparse
from . import Encoder

def process_json_file(input_path, output_path, model):
    """
    Process a JSON file to generate embeddings for each complete entry.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        model: Encoder model instance
    """
    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Generate embeddings for each complete entry
    embeddings = {}
    for entry in data:
        entry_str = json.dumps(entry)  # Convert entire entry to string
        embeddings[entry['id']] = model.encode(entry_str)

    # Save embeddings to output file
    with open(output_path, 'w') as f:
        json.dump(embeddings, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for text files.")
    parser.add_argument("input_path", help="Path to the input folder containing text files")
    parser.add_argument("output_path", help="Path to the output JSON file")
    parser.add_argument("--model", default="text-embedding-3-small", help="Model to use for embedding generation")
    args = parser.parse_args()


    try:
        model = Encoder(args.model)
        input_filename = os.path.splitext(os.path.basename(args.input_path))[0]  # For the directory name
        output_name = f"{model.name}.json"
        os.makedirs(os.path.join(args.output_path, input_filename), exist_ok=True) # make sure the directory exists

        output_path = os.path.join(args.output_path, input_filename, output_name)
        process_json_file(args.input_path, output_path, model)
    except Exception as e:
        print(f"Error: {e}")
