import os

from src import process_text_files
from src import DATA_DIR

def test_generate_embeddings():
    input_folder = os.path.join(DATA_DIR, 'txts', 'test')
    output_file = os.path.join(DATA_DIR, 'emb', 'test', 'test_embeddings.json')
    process_text_files(input_folder, output_file)

if __name__ == '__main__':
    test_generate_embeddings()