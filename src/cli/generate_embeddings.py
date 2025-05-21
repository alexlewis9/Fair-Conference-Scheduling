import argparse
from src.data_processing.generate_embeddings import generate_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text files.")
    parser.add_argument("input_path", help="Path to the input folder containing text files")
    parser.add_argument("output_path", help="Path to the output JSON file")
    parser.add_argument("--model", default="text-embedding-3-small", help="Model to use for embedding generation")
    parser.add_argument('--include', type=lambda s: s.split(','),
                        default=[], help='Comma-separated list of keys at each entry to include')
    parser.add_argument('--exclude', type=lambda s: s.split(','),
                        default=[], help='Comma-separated list of keys at each entry to exclude')
    args = parser.parse_args()
    try:
        generate_embeddings(args.input_path, args.output_path, args.model, include=args.include, exclude=args.exclude)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()