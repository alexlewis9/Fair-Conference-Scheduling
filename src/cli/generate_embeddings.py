import argparse
from src.data_processing.generate_embeddings import generate_embeddings
from src.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text files.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    try:
        generate_embeddings(cfg['input_path'], cfg['output_path'], cfg['model'], include=cfg['include'], exclude=cfg['exclude'], stride=cfg['stride'])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()