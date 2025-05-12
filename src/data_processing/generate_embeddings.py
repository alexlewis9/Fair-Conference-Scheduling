import os
from openai import OpenAI
import numpy as np
import json

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """
    Get the embedding for a given text using OpenAI's API.
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

def process_text_files(input_folder, output_file):
    """
    Processes all text files in the input folder and generates embeddings for each file.
    Saves the embeddings to a JSON file.
    """
    embeddings = {}
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    embeddings[os.path.basename(file_path)] = get_embedding(text)
    with open(output_file, "w") as f:
        json.dump(embeddings, f)

if __name__ == "__main__":
    input_folder = ".//txts"
    output_file = "embeddings.json"
    process_text_files(input_folder, output_file)
    print(f"Embeddings saved to {output_file}")